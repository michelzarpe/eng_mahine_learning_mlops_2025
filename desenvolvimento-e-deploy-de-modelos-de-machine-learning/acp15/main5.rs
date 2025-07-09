// Compartilhamento Seguro de Dados Entre Múltiplas Threads

// Importando Arc (Atomic Reference Counted) e Mutex da biblioteca padrão para compartilhamento seguro de dados entre threads
use std::sync::{Arc, Mutex};

// Importando o módulo thread da biblioteca padrão para manipulação de threads
use std::thread;

fn main() {

    // Criando um contador protegido por um Mutex e compartilhado entre threads usando Arc
    let counter = Arc::new(Mutex::new(0));

    // Criando um vetor para armazenar os handles das threads
    // Em Rust, um "handle" refere-se a uma instância que representa a "propriedade" ou o "controle" de um recurso, como uma thread. 
    let mut handles = vec![];

    // Loop para criar 10 threads
    for _ in 0..10 {

        // Clonando o Arc para compartilhar o contador entre as threads
        let counter = Arc::clone(&counter);

        // Criando uma nova thread
        let handle = thread::spawn(move || {

            // Obtendo um lock no Mutex e acessando o contador
            let mut num = counter.lock().unwrap();

            // Incrementando o contador
            *num += 1;
        });

        // Adicionando o handle da thread ao vetor
        handles.push(handle);
    }

    // Esperando todas as threads terminarem
    for handle in handles {
        handle.join().unwrap();
    }

    // Imprimindo o valor final do contador
    println!("Contagem Final: {}", *counter.lock().unwrap());
}
