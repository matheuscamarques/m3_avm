//! tui.rs — REPL de terminal para o modo `--real --interactive`.
//!
//! Problema que resolve: com eco canônico do tty, o que o usuário digita e o
//! streaming da LLM saem na mesma linha e se misturam. Aqui a tela tem duas
//! áreas lógicas mantidas por invariante:
//!
//! ```text
//! ...histórico (rola normal)...
//! <linha de saída atual, cresce com o streaming>
//! > <linha de input, sempre por último>
//! ```
//!
//! - `print_output(chunk)`: apaga a linha de input, sobe 1, anexa `chunk` no
//!   fim da linha de saída, desce para linha nova e redesenha `> buf`.
//!   Chunks sem `\n` continuam a linha de saída; mensagens devem terminar
//!   com `\n` (ver [`Repl::print_line`]).
//! - Input em modo raw (sem eco/ICANON/ISIG via `libc`, já dependência):
//!   lemos cada tecla e redesenhamos `> buf` nós mesmos. Ctrl+C vira
//!   [`Input::Interrupt`] (sem sinal, sem sujeira no terminal).
//! - Sem tty (pipe `echo ... |`, CI): modo plain — mesmo comportamento de
//!   antes (thread de stdin + `print!` direto), zero sequência ANSI.
//!
//! Restauração do terminal é via `Drop` (RAII); como ISIG fica desligado,
//! Ctrl+C não mata o processo no meio de uma sequência ANSI.

use std::io::{self, Write};

/// Evento de input (espelha `mpsc::TryRecvError` + Ctrl+C explícito).
#[derive(Debug)]
pub enum Input {
    Line(String),
    Empty,
    Eof,
    Interrupt,
}

const PROMPT: &str = "> ";

pub struct Repl {
    tty: bool,
    buf: Vec<u8>,
    staging: Vec<u8>,
    plain_rx: Option<std::sync::mpsc::Receiver<String>>,
    _guard: Option<TermGuard>,
}

impl Repl {
    /// Detecta tty; em tty entra em raw mode e prepara a área de input.
    pub fn new() -> io::Result<Self> {
        // SAFETY: isatty(0) só lê estado do fd.
        let tty = unsafe { libc::isatty(0) == 1 };
        if !tty {
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                let mut line = String::new();
                loop {
                    line.clear();
                    match stdin.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        let _ = tx.send(trimmed);
                    }
                }
            });
            return Ok(Self { tty: false, buf: Vec::new(), staging: Vec::new(), plain_rx: Some(rx), _guard: None });
        }
        let guard = TermGuard::new()?;
        let mut r = Self { tty: true, buf: Vec::new(), staging: Vec::new(), plain_rx: None, _guard: Some(guard) };
        // Garante que existe uma linha de saída vazia acima da linha de input.
        r.write_raw("\n")?;
        r.render_input()?;
        Ok(r)
    }

    pub fn is_tty(&self) -> bool {
        self.tty
    }

    /// Chunk de streaming (sem `\n` no fim continua a linha de saída).
    pub fn print_output(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.tty {
            print!("{text}");
            return io::stdout().flush();
        }
        let clean = text.replace('\r', "");
        // (1) limpa linha de input (2) sobe p/ linha de saída (3) vai ao fim
        // (4) escreve (5) desce p/ linha nova (6) redesenha input.
        self.write_raw("\r\x1b[K\x1b[1A\x1b[999C")?;
        self.write_raw(&clean)?;
        self.write_raw("\n")?;
        self.render_input()
    }

    /// Mensagem de linha (status, rollback, EOS...): sempre termina a linha.
    pub fn print_line(&mut self, text: &str) -> io::Result<()> {
        if text.ends_with('\n') {
            self.print_output(text)
        } else {
            let mut s = String::with_capacity(text.len() + 1);
            s.push_str(text);
            s.push('\n');
            self.print_output(&s)
        }
    }

    /// Não-bloqueante: consome teclas pendentes; ENTER submete a linha.
    pub fn try_poll(&mut self) -> Input {
        if !self.tty {
            return match &self.plain_rx {
                Some(rx) => match rx.try_recv() {
                    Ok(line) => Input::Line(line),
                    Err(std::sync::mpsc::TryRecvError::Empty) => Input::Empty,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => Input::Eof,
                },
                None => Input::Eof,
            };
        }
        // Drena bytes disponíveis para staging (non-blocking).
        // read==0 significa EOF real; read<0 é EAGAIN (sem dados).
        let mut tmp = [0u8; 256];
        let mut eof = false;
        loop {
            // SAFETY: read(0) em fd válido; tmp dimensionado.
            let n = unsafe { libc::read(0, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len()) };
            if n > 0 {
                self.staging.extend_from_slice(&tmp[..n as usize]);
                if (n as usize) < tmp.len() {
                    break;
                }
            } else if n == 0 {
                eof = true;
                break;
            } else {
                break;
            }
        }
        if self.staging.is_empty() {
            return if eof { Input::Eof } else { Input::Empty };
        }
        let bytes: Vec<u8> = std::mem::take(&mut self.staging);
        let mut rest = Vec::with_capacity(bytes.len());
        let mut i = 0;
        let mut line_out: Option<Input> = None;
        while i < bytes.len() {
            let b = bytes[i];
            match b {
                // ENTER: submete (eco da linha vai p/ histórico + nova área input)
                b'\r' | b'\n' => {
                    // Trata \r\n como um ENTER só
                    if b == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                        i += 1;
                    }
                    let line = String::from_utf8_lossy(&self.buf).trim().to_string();
                    self.buf.clear();
                    // Linha submetida entra no histórico; recria área de input.
                    let _ = self.write_raw("\r\x1b[K");
                    if !line.is_empty() {
                        let _ = self.write_raw(PROMPT);
                        let _ = self.write_raw(&line);
                    }
                    let _ = self.write_raw("\n\n");
                    let _ = self.render_input();
                    line_out = Some(if line.is_empty() { Input::Empty } else { Input::Line(line) });
                    // bytes restantes ficam p/ próxima poll
                    rest.extend_from_slice(&bytes[i + 1..]);
                    break;
                }
                // Ctrl+C → interrupção graciosa (ISIG desligado: chega como byte)
                0x03 => {
                    line_out = Some(Input::Interrupt);
                    rest.extend_from_slice(&bytes[i + 1..]);
                    break;
                }
                // Ctrl+D em linha vazia → EOF
                0x04 if self.buf.is_empty() => {
                    line_out = Some(Input::Eof);
                    rest.extend_from_slice(&bytes[i + 1..]);
                    break;
                }
                // Backspace: apaga 1 char (respeita UTF-8)
                0x7f | 0x08 => {
                    while !self.buf.is_empty() {
                        self.buf.pop();
                        if String::from_utf8(self.buf.clone()).is_ok() {
                            break;
                        }
                    }
                    let _ = self.render_input();
                }
                // ESC: consome sequência de escape (setas etc.) e ignora
                0x1b => {
                    i += 1;
                    if bytes.get(i) == Some(&b'[') {
                        i += 1;
                        while i < bytes.len() && !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'~') {
                            i += 1;
                        }
                        i += 1; // consome o byte final
                    }
                    continue;
                }
                // Ctrl+U: limpa linha
                0x15 => {
                    self.buf.clear();
                    let _ = self.render_input();
                }
                // NUL / Ctrl+Z: o pty pode entregar \x00 (EOF do master);
                // com ISIG desligado, Ctrl+Z chega como byte — ignora ambos.
                0x00 | 0x1a => {}
                _ => {
                    self.buf.push(b);
                    let _ = self.render_input();
                }
            }
            i += 1;
        }
        self.staging = rest;
        match line_out {
            Some(ev) => ev,
            None if eof => Input::Eof,
            None => Input::Empty,
        }
    }

    fn render_input(&mut self) -> io::Result<()> {
        debug_assert!(self.tty);
        let mut out = io::stdout().lock();
        out.write_all(b"\r\x1b[K")?;
        out.write_all(PROMPT.as_bytes())?;
        out.write_all(&self.buf)?;
        out.flush()
    }

    fn write_raw(&mut self, s: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        out.write_all(s.as_bytes())?;
        out.flush()
    }
}

/// Guarda RAII: raw mode + non-blocking no fd 0; restaura no Drop.
struct TermGuard {
    orig_term: libc::termios,
    orig_flags: libc::c_int,
}

impl TermGuard {
    fn new() -> io::Result<Self> {
        // SAFETY: fds/fcntl/termios em fd 0 válido (é tty, checado antes).
        unsafe {
            let mut orig_term: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut orig_term) != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut raw = orig_term;
            // cfmakeraw manual: sem eco, canônico ou sinais; mantém OPOST.
            raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::ISIG | libc::IEXTEN);
            raw.c_iflag &= !(libc::IXON | libc::ICRNL | libc::BRKINT | libc::INPCK | libc::ISTRIP);
            raw.c_cc[libc::VMIN as usize] = 1;
            raw.c_cc[libc::VTIME as usize] = 0;
            if libc::tcsetattr(0, libc::TCSANOW, &raw) != 0 {
                return Err(io::Error::last_os_error());
            }
            let orig_flags = libc::fcntl(0, libc::F_GETFL);
            if orig_flags < 0 {
                libc::tcsetattr(0, libc::TCSANOW, &orig_term);
                return Err(io::Error::last_os_error());
            }
            if libc::fcntl(0, libc::F_SETFL, orig_flags | libc::O_NONBLOCK) != 0 {
                libc::tcsetattr(0, libc::TCSANOW, &orig_term);
                return Err(io::Error::last_os_error());
            }
            Ok(Self { orig_term, orig_flags })
        }
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        // SAFETY: restaura estado original; ignora erros (já estamos saindo).
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &self.orig_term);
            libc::fcntl(0, libc::F_SETFL, self.orig_flags);
        }
        // Garante cursor na próxima linha ao sair.
        let _ = io::stdout().write_all(b"\r\n");
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_plain_passthrough_without_tty() {
        // Sem tty (cargo test roda com stdin fechado/redirecionado), Repl cai
        // em modo plain e try_poll nunca bloqueia.
        if unsafe { libc::isatty(0) == 1 } {
            eprintln!("skip: rodando com tty");
            return;
        }
        let mut r = Repl::new().unwrap();
        assert!(!r.is_tty());
        match r.try_poll() {
            Input::Empty | Input::Eof => {}
            Input::Line(_) => {}
            Input::Interrupt => {}
        }
        r.print_output("hello").unwrap();
    }
}
