// Manipulação de Arquivos

// Importando o módulo File da biblioteca padrão para manipulação de arquivos
use std::fs::File;

// Importando módulos do io (entrada/saída) para operações de leitura e tratamento de erros
use std::io::{self, Read};

// Importando o tipo ParseIntError para tratamento de erros de parse
use std::num::ParseIntError;

// Função que lê o conteúdo de um arquivo e retorna a variável Result com o conteúdo como String
fn dsa_read_file(filename: &str) -> Result<String, io::Error> {
    
    // Abre o arquivo especificado por filename
    let mut file = File::open(filename)?;
    
    // Cria uma string para armazenar o conteúdo do arquivo
    let mut contents = String::new();
    
    // Lê o conteúdo do arquivo para a string
    file.read_to_string(&mut contents)?;
    
    // Retorna o conteúdo do arquivo como Ok
    Ok(contents)
}

// Função que faz parse de uma string para um número inteiro de 32 bits
fn dsa_parse_string_numero(s: &str) -> Result<i32, ParseIntError> {

    // Remove espaços em branco da string e tenta convertê-la para i32
    s.trim().parse::<i32>()
}

// Função principal que retorna um Result vazio em caso de sucesso ou um erro em caso de falha
fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // Nome do arquivo a ser lido
    let filename = "src/numero.txt";
    
    // Lê o conteúdo do arquivo
    let contents = dsa_read_file(filename)?;
    
    // Faz parse do conteúdo lido para um número
    let number = dsa_parse_string_numero(&contents)?;
    
    // Imprime o número
    println!("O número é: {}", number);
    
    // Retorna Ok se tudo ocorrer sem erros
    Ok(())
}
