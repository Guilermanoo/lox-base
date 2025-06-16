

Já que estamos lidando com uma pilha de dicionários, onde cada dicionário 
representa um nível de escopo, vale a pena definir os métodos abaixo:

```python
class Ctx:
    ...
    def pop(self) -> dict[str, Value]:
        "Remove o topo da pilha, retornando-o"
        # implemente!

    def push(self, tos: dict[str, Value]):
        "Adiciona um dicionário ao topo da pilha"
        # implemente!
```

Implemente esses métodos.

Finalmente, com essa nova estrutura implementada, podemos melhorar o controle
de escopo em duas partes importantes do nosso interpretador. Primeiro, na 
implementação do método `Block.eval` e depois de `LoxFunction.eval`.

Em ambos casos, chamamos ctx.push({}) com um contexto vazio no início do método
e terminamos com um ctx.pop() para descartar esse último nível de contexto ao 
sair da função.

```python
class Block(Stmt):
    ...
    def eval(self, ctx: Ctx):
        ctx.push({})
        # a implementação original entra aqui!
        ctx.pop()
```

Em LoxFunction precisamos de um pouco mais de cuidado por causa da interação com 
as exceções LoxReturn:

```python
class LoxFunction(Stmt):
    ...
    def call(self, *args):
        self.ctx.push({})
        try:
            # a implementação original entra aqui!
        finally:
            ctx.pop()
```

Ainda tem um pequeno detalhe antes de falarmos que a implementação da resolução 
de escopos estava completa. Em Lox, diferenciamos entre declaração de variáveis
(`var x = 42`) e uma atribuição simples (`x = 42`, sem o var). No primeiro caso,
criamos uma nova variável no último nível de escopo. Essa é a lógica implementada
no nosso `Ctx.__setitem__`. Já o segundo caso é um pouco mais trabalhoso: devemos
buscar na pilha o primeiro dicionário que guarda a variável x e mudar seu valor
apenas nesse dicionário. Caso não exista em lugar nenhum, devemos levantar um
erro ao invés de criar uma variável no topo da pilha.

```lox
// ctx = [{"x": 1}]
var x = 1;
{
    // ctx = [{"x": 1}, {"y": 2}]
    var y = 2;
    {
        // ctx = [{"x": 3}, {"y": 2}, {}]
        //         ^ modifica esse escopo
        x = 3;
        // erro, porque z não foi encontrado em nenhum nível do escopo.
        z = 4;
    }
}
```

Para acomodar esse comportamento, implementamos o método assign, para atribuir
um valor a uma variável existente:

```python
class Ctx:
    ...
    def assign(self, key: str, value: Value):
        """
        Encontra a ocorrência de key mais próxima do topo da pilha e 
        troca seu valor por value.
        """
        ... # implemente!
```

Com isso, devemos modificar `Assign.eval` para usar `ctx.assign(name, value)` ao
invés de `ctx[name] = value`, como fazíamos antes. Pronto! Agora nossa 
implementação das regras de escopo deve estar completa!
