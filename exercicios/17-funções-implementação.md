[dificil]

Vamos agora implementar o suporte a funções. A estratégia para isso será criar uma
classe que guarde as definções importantes para o funcionamento da função: o seu nome, 
lista de argumentos a árvore sintática que declara o corpo da função e o contexto de 
execução de quando a função foi declarada.

Vamos começar montando a estrutura básica da nossa estratégia criando uma classe
no arquivo lox/runtime.py da seguinte forma:

```python
@dataclass
class LoxFunction:
    name: str
    params: list[str]
    body: Expr
    ctx: Ctx

    def call(self, args: list[Value]):
        ...
```

Nosso principal trabalho aqui é implementar o método call, que recebe uma lista de
argumentos e executa o corpo da função. O primeiro passo da execução do contexto
é receber a lista de argumentos e salvá-las no contexto de execução. Para isso,
precisamos associar cada valor em args com seu nome correspondente em params,
salvando-os no dicionário de contexto.

Seria um código mais ou menos assim:

```python
for param_name, arg in zip(params, args):
    ctx[param_name] = arg
```

Faça as adaptações necessárias para esse código funcionar na classe.

Em seguida, usamos esse contexto devidamente atualizado para rodar o corpo da 
função. Essa parte é fácil, basta fazer

```python
body.eval(ctx)
```

Note que o resultado de `body.eval(ctx)` não corresponde necessariamente ao valor 
de retorno da função. Em Lox (como em várias outras linguagens), o valor de retorno
deve ser fornecido pela palavra reservada `return`. O livro faz uma 
discussão interessante
sobre as dificuldades de identificar a localização deste return. Aqui vamos usar
a mesma estratégia de implementar o return levantando exceções.

Para isso, leia a seção sobre [Return Statements]((https://craftinginterpreters.com/functions.html#return-statements))
no livro para entender a abordagem geral. Em Python vamos criar a classe 

```python
class LoxReturn(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value
```

E devemos trocar nosso comando `body.eval(ctx)` por algo que lide com as execções
de LoxReturn, caso elas aconteçam,

```python
try:
    body.eval(ctx)
except LoxReturn as ex:
    return ex.value
```

Para que isso funcione, temos que modificar a implementação de eval nos nós de
return para que levantem uma exceção `LoxReturn(value)` com o valor desejado ao
invés de retorná-lo como em `return value`.

Para finalizar nossa implementação, podemos fornecer implementar o método `__call__`
para fazer com que as nossas LoxFunctions se comportem essencialmente como funções
Python e possam ser chamadas como se tivessem sido definidas por um `def` no
código Python.

```python

class LoxFunction:
    ...
    def __call__(self, *args):
        return self.call(args)
```

Dessa forma, não precisamos mudar a definição do método eval para os nós de `Call`,
já que as funções Lox também se comportarão como funções Python arbitrárias. Podemos
inclusive criar funções em Lox e integrá-las no nosso código Python de forma 
transparente, como se tivessem sido definidas em Python normalmente. 

Já temos um sistema de mão dupla entre Python e Lox: podemos passar funções Python
para Lox ao configurar o contexto inicial antes de carregar o arquivo .lox e
também podemos usar essas funções do Lox em Python lendo-as no dicionário de 
contexto e usando-as do Python como uma função qualquer!