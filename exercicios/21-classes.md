[médio]

LoxClass é uma classe Python cujas instâncias representam classes Lox. Também
criaremos uma segunda classe LoxInstance que representa objetos Lox em tempo de
execução.

As instâncias de LoxClass são objetos que representam classes Lox. Esses objetos
são responsáveis por duas funcionalidades importantes: criar instâncias de
LoxInstance e encontrar os métodos apropriados para uma instância.

Nessa atividade, implementaremos essas duas funcionalidades. 

## Construtor

O método `Class.eval` por enquanto cria LoxClasse's sem passar nenhum
argumento para o seu construtor. Isso obviamente não funciona, já que a classe 
deve ter acesso aos detalhes de sua implementação. Reescreva o seu LoxClass
para ter uma estrutura parecida com a abaixo:

```python
@dataclass
class LoxClass:
    name: str
    methods: dict[str, LoxFunction]
    base: Optional["LoxClass"]
```

## Métodos

Agora, implemente os dois métodos importantes da nossa LoxClass:

```python
class LoxClass:
    ...

    def __call__(self, *args):
        """
        self.__call__(x, y) <==> self(x, y)

        Em Lox, criamos instâncias de uma classe chamando-a como uma função. É
        exatamente como em Python :)
        """

        # Por enquanto, retornamos instâncias genéricas
        return LoxInstance()

    def get_method(self, name: str) -> "LoxFunction":
        # Procure o método na classe atual. 
        # Se não encontrar, procure nas bases.
        # Se não existir em nenhum dos dois lugares, levante uma exceção do
        # tipo LoxError.

        ... # sua implementação aqui!
        return method
```

Agora que implementamos as funcionalidades básicas da nossa LoxClass, vamos para
a segunda parte da nossa atividade e implementar a função `Class.eval`
corretamente. Ela é responsável por fazer algumas coisas importantes:

* Deve converter a lista com as árvores sintáticas de todos os métodos em um
  dicionário mapeando o nome do método em sua implementação como uma
  `LoxFunction`.
* Deve encontrar a classe base no contexto, caso ela tenha sido declarada.
* Finalmente, deve construir o objeto LoxClass e salvá-lo no contexto.

A estrutura geral está delineada abaixo

```python
@dataclass
class Class(Stmt):
    ...

    def eval(self, ctx: Ctx):
        # Carrega a superclasse, caso exista
        superclass = get_superclass(ctx)  # verifique no contexto e retorne a base ou None
        class_name = ...      # nome da classe
        method_defs = ...     # definição dos métodos
        
        # Avaliamos cada método
        methods = {}
        for method in method_defs:
            # não podemos simplesmente avaliar method.eval(ctx) porque isso 
            # acrescentaria { method.name: method_impl } ao contexto de execução.
            method_name = ...
            method_body = ...
            method_args = ...
            method_impl = LoxFunction(method_name, method_args, method_body, ctx)
            methods[method_name] = method_impl

        lox_class = LoxClass(class_name, methods, superclass)
        ctx.var_def(ctx)
        return lox_class
```

Complete esta implementação e veja os testes passarem :)
