// Fundamentos da Linguagem Rust

// Importa a funcionalidade de geração de números aleatórios do pacote `rand`.
use rand::Rng;

// Importa a funcionalidade para comparar valores.
use std::cmp::Ordering;

// Importa a biblioteca de entrada/saída do Rust.
use std::io;

// Função principal que é o ponto de entrada do programa.
fn main() {

    // Imprime uma mensagem de boas-vindas na tela.
    println!("Vamos Jogar Um Jogo?");

    // Gera um número aleatório entre 1 e 100 e armazena na variável `secret_number`.
    let secret_number = rand::thread_rng().gen_range(1..=100);

    // Mostra o número secreto na tela. Útil para depuração.
    println!("O número secreto é: {}", secret_number);

    // Pede ao usuário para adivinhar o número secreto.
    println!("Consegue Adivinhar o Número Secreto? Digite Um Número:");

    // Declara uma variável mutável `dsanum` para armazenar a entrada do usuário.
    let mut dsanum = String::new();

    // Lê uma linha do terminal e armazena na variável `dsanum`.
    io::stdin()
        .read_line(&mut dsanum)
        .expect("Falha ao ler o input");

    // Tenta converter a entrada em um número inteiro de 32 bits.
    let dsanum: u32 = dsanum.trim().parse().expect("Digite Um Número!");

    // Compara o palpite com o número secreto e imprime uma mensagem correspondente.
    match dsanum.cmp(&secret_number) {
        
        // Caso o palpite seja menor que o número secreto, informa ao usuário que seu palpite foi baixo.
        Ordering::Less => println!("Você digitou um número menor que o número secreto!"),
        
        // Caso o palpite seja maior que o número secreto, informa ao usuário que seu palpite foi alto.
        Ordering::Greater => println!("Você digitou um número maior que o número secreto!"),
        
        // Caso o palpite seja igual ao número secreto, parabeniza o usuário por ganhar.
        Ordering::Equal => println!("Você Adivinhou o Número. Você Venceu!!!"),
    }
}
