[medio]

No exercício passado implementando a atribuição de variáveis, que 
corresponde à parte sem o (call ".")? da regra abaixo

    assignment  → ( call "." )? IDENTIFIER "=" assignment
                | logic_or ;

A produção `call "." IDENTIFIER "=" assignment` corresponde à redefinção
de atribuitos. No caso, call pode ser um objeto arbitrário possivelmente
armazenado em uma variável ou produzido por uma chamada de função arbitrária.

Diferentemente do `Assign` do exercício anterior, esta regra requer 3 
argumentos e por isso vamos dedicar uma classe separada à ela. Chamamos de 
`Setattr` em paralelo à classe `Getattr` definida anteriormente.

Implemente a atribuição de atributos na nossa gramática. Lembre-se de usar 
a função `setattr(obj, attr_name, value)` nativa do Python para atribuir um
atributo dinamicamente.