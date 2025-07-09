// Manipulação de Arquivos

// Importando o módulo File da biblioteca padrão para manipulação de arquivos
use std::fs::{File};

// Importando módulos do io (entrada/saída) para operações de escrita, leitura e tratamento de erros
use std::io::{self, Write, Read};

// Importando o módulo fs para operações no sistema de arquivos
use std::fs;

// Função que cria um arquivo e grava o conteúdo especificado
fn dsa_cria_grava_arquivo(filename: &str, content: &str) -> io::Result<()> {

    // Cria ou substitui o arquivo especificado por filename
    let mut file = File::create(filename)?;

    // Escreve todo o conteúdo fornecido no arquivo
    file.write_all(content.as_bytes())?;

    // Retorna Ok se a operação for bem-sucedida
    Ok(())
}

// Função que lê o conteúdo de um arquivo e retorna como uma String
fn dsa_le_arquivo(filename: &str) -> io::Result<String> {

    // Abre o arquivo especificado por filename
    let mut file = File::open(filename)?;

    // Cria uma string para armazenar o conteúdo do arquivo
    let mut contents = String::new();

    // Lê o conteúdo do arquivo para a string
    file.read_to_string(&mut contents)?;

    // Retorna o conteúdo do arquivo como Ok
    Ok(contents)
}

// Função que deleta o arquivo especificado
fn dsa_deleta_arquivo(filename: &str) -> io::Result<()> {

    // Remove o arquivo especificado por filename
    fs::remove_file(filename)?;

    // Retorna Ok se a operação for bem-sucedida
    Ok(())
}

// Função principal que executa operações de criação, leitura e deleção de arquivos
fn main() -> io::Result<()> {

    // Nome do arquivo a ser manipulado
    let filename = "src/exemplo.txt";
    
    // Conteúdo a ser escrito no arquivo
    let content = "Hello, Aprendendo Rust com a DSA!";

    // Cria o arquivo e grava o conteúdo
    dsa_cria_grava_arquivo(filename, content)?;
    println!("Arquivo criado e conteúdo gravado.");

    // Lê o conteúdo do arquivo
    let contents = dsa_le_arquivo(filename)?;
    println!("Conteúdo do arquivo: {}", contents);

    // Deleta o arquivo
    dsa_deleta_arquivo(filename)?;
    println!("Arquivo deletado.");

    // Retorna Ok se todas as operações forem bem-sucedidas
    Ok(())
}


/*

A interrogação (?) em Rust é usada para simplificar a manipulação de erros. 

Ela é uma forma concisa de escrever código que pode falhar e retornar um erro. Quando usada, a interrogação verifica se a operação resultou em um erro. 

Se sim, o erro é retornado da função que contém o operador ?. Se não, o valor resultante da operação bem-sucedida é desembrulhado e retornado.

Por exemplo: No caso específico de let mut file = File::create(filename)?;, o operador ? serve para:

- Tentar criar o arquivo: File::create(filename) tenta criar um arquivo com o nome especificado.
- Verificar o resultado: Se a operação for bem-sucedida, o valor (que é um File) é atribuído à variável file.
- Propagar o erro: Se a operação falhar, o erro (std::io::Error) é retornado da função que contém essa linha de código, propagando o erro para o chamador da função.

Assim, o operador ? simplifica o código ao eliminar a necessidade de escrever manualmente o controle de fluxo para verificar e propagar erros.

*/




