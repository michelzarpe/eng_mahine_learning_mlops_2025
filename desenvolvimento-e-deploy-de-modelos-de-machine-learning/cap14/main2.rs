// Fundamentos da Linguagem Rust

// Importa a biblioteca de entrada/saída do Rust.
use std::io;

// Importa a biblioteca rand para geração de números aleatórios.
use rand::Rng;

// Função principal que é o ponto de entrada do programa.
fn main() {
    
    // Imprime uma mensagem na tela.
    println!("Vamos Jogar Um Jogo?");

    // Gera um número aleatório entre 1 e 100 e armazena em `secret_number`.
    let secret_number = rand::thread_rng().gen_range(1..=100);

    // Imprime o número secreto - útil para depuração.
    println!("O número secreto é: {}", secret_number);

    // Solicita ao usuário para digitar um número.
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
