[medio]

Em Lox, declaramos funções usando a sintaxe

```lox
fun f(x, y, x) {
    // implementação da função
}
```
    
Isto está descrito em detalhes no capítulo sobre
[declaração de funções](https://craftinginterpreters.com/functions.html#function-declarations)

Nesse exercício, você deve implementar o suporte à declaração de funções na
gramática. Para isso, acrescente as regras de declaração de função seguindo a
especificação do livro.

A parte mais difícil de implementar o suporte a funções na linguagem está na
lógica de execução, ou seja, na implementação do método `Function.eval()`. O
método eval deveria construir uma função Python (ou algo equivalente) e salvá-la
no contexto de execução. Vamos deixar esse problema pro futuro e, por enquanto,
podemos criar uma implementação falsa, apenas para passar nos testes:

```python
@dataclass
class Function(Stmt)
    name: str
    ...

    def eval(self, ctx: Ctx):
        # Nossa função retorna 42, independente do que ela deveria fazer de
        # acordo com o código Lox
        def function(*args):
            return 42

        ctx[self.name] = function
        return function
```

Os testes para esse exercício sempre chamam funções que avaliam para o mesmo 
resultado 42, de forma que continuarão funcionando quando implementarmos nossas 
funções corretamente.