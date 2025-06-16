[dificil]

O [laço for](https://craftinginterpreters.com/control-flow.html#for-loops) de Lox
é semelhante ao de C. Em comparação ao `while`, o `for` possui uma sintaxe
consideravelmente mais complexa, já que declaramos um inicializador, condição e 
incrementador ao invés da condição simples no laço `while`. Além do mais, todas
essas condições podem ser omitidas. 

O `for` em Lox (assim como em outras linguagens de programação) é uma inclusão
estritamente desnecessária à linguagem: todo laço `for` pode ser reescrito como um
`while`. Ele existe somente por conveniência do programador e por ser, de modo
geral, uma forma de declarar laços um pouco menos sujeita a erros.

Vamos aproveitar isso no nosso interpretador e, ao invés de criar nós do tipo
`lox.ast.For`, vamos usar o LoxTransformer para transformar todas as ocorrências
de for no nó `while` correspondente. A regra de conversão em Lox é simples:

    for (init; cond; incr) stmt => { init; while (cond) {stmt; incr} }

Chamamos expressões deste tipo (que podem ser facilmente convertidas em outras mais básicas)
de "açúcar sintático".

Implemente as funções que lidam com o laço for no LoxTransformer e faça-o
retornar um nó do tipo Block com a estrutrura descrita acima. Depois apague a
classe `lox.ast.For` do arquivo lox/ast.py, porque ela não será mais necessária.

## Observação

O fato de que tanto `init`, como `cond` como `incr` serem opcionais dificulta um
pouco como lidar com laços for no `LoxTransformer`. A regra

    forStmt → "for" "(" ( varDecl | exprStmt | ";" ) expression? ";" expression? ")" statement ;

pode produzir de 1 a 4 filhos dependendo de quais partes opcionais foram
ativadas. Deste modo, nosso transformer não conseguiria diferenciar algumnas
declarações diferentes entre si. Por exemplo, tanto `for (;x;) {}` quanto
`for (;;x) {}` possuem dois filhos, apesar de representarem códigos bastante
diferentes entre si (ambos possivelmente ruins, já podemos adiantar).

Para lidar com isso, recomendo decompor a regra sintática para o `for` mais ou
menos assim:

    forStmt → "for" "(" forInit forCond ";" forIncr ")" statement ;
    forInit → ( varDecl | exprStmt | ";" )
    forCond → expression?
    forIncr → expression?

Deste modo, o transformer receberá sempre o mesmo número de argumentos, ainda
que alguns destes possam ser None.

Caso estejam vazios, podemos considerar a seguinte semântica:

    forInit => vira um Literal(None)
    forCond => vira um Literal(True)
    forIncr => vira um Literal(None)

