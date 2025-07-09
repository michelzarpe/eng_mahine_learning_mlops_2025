// Fundamentos da Linguagem Rust

// Importa a biblioteca de entrada/saída do Rust.
use std::io;

// Função principal que é o ponto de entrada do programa.
fn main() {

    // Imprime uma mensagem na tela.
    println!("Vamos Jogar Um Jogo?");

    // Pede ao usuário para digitar um número.
    println!("Digite Um Número:");

    // Cria uma variável mutável para armazenar a entrada do usuário como texto.
    let mut dsanum = String::new();

    // Lê uma linha do terminal e armazena na variável `dsanum`.
    io::stdin()
        .read_line(&mut dsanum)
        .expect("Falha ao ler o input");

    // Imprime o valor que o usuário digitou.
    println!("Você Digitou: {}", dsanum);
}
