[difícil]

Na atividade anterior, completamos a implementação de LoxInstance para que os
objetos Lox consigam fazer operações básicas como guardar atributos e chamar
métodos. Existe uma limitação importante na nossa implementação, que é o fato
que os nossos métodos ainda não suportam o uso de `this`. Em linguagem de OO, é
como se todos os métodos fossem estáticos.

A estratégia para consertar nossa implementação envolve duas etapas:

1. Suporte sintático ao `this`: por enquanto, nossa gramática trata o this como
   uma variável comum. Você pode testar isso executando
   `uv run lox arquivo.lox --ast` em algum `arquivo.lox` que use a palavra
   reservada `this`. 
2. Implementar suporte ao this em LoxFunction. Em qualquer linguagem orientada a
   objetos, métodos são funções ordinárias que inserem an instância no escopo.
   No caso do Lox isso é feito de forma implícita expondo o a instância como uma
   variável especial chamada `this`. (no Python, por outro lado, a instância é
   declarada explicitamente como um dos argumentos do método que, por convenção,
   chamamos de `self` mas poderia ter qualquer outro nome.)

Tecnicamente nem sequer precisaríamos nos preocupar com o primeiro problema, já
que é possível tratar `this` como uma variável comum que possui a restrição de
aparecer apenas em métodos de uma classe. No entanto, em atividades futuras será
interessante distinguir `This` de `Var` para efeitos da análise semântica. Minha
sugestão é simplesmente copiar a implementação de `Var`, incluindo o método
eval, na classe `This`. Assim, a semântica de `this` vai ser "procure uma
variável chamada `"this"` no contexto atual de execução e retorne seu valor".

O segundo ponto é mais complicado um pouco. Devemos acrescentar this ao contexto
de execução dos métodos. O livro faz uma discussão detalhada das possiblidades
em https://craftinginterpreters.com/classes.html#this. Vamos seguir uma
estratégia semelhante e pensar que um método = função + um escopo que associa o
nome `this` à instância chamando o método.

Vamos chamar essa operação de associar a função declarada no corpo da classe à
uma instância de `.bind()`. `LoxFunction.bind(instance)` deve criar uma cópia de
LoxFunction com a única diferença que o contexto carrega um escopo adicional com
`{"this": instance}`. Assim, quando a função for executada, qualquer referência
à `this` será resolvida para a instância associada ao método.

```python
class LoxFunction:
    ... 

    def bind(self, obj: Value) -> "LoxFunction":
        # Associamos essa cópia a um this específico
        return LoxFunction(
            ..., # outros argumentos aqui, na ordem correta.
            ctx.push({"this": obj})
        )
```

Agora que sabemos associar funções a um `this` específico, podemos modificar a
função `LoxInstance.__getattr__` para associar os métodos à instância antes de
retorná-los.

```python
class LoxInstance:
    ...
    
    def __getattr__(self, attr: str) -> "LoxFunction":
        ... # lógica original
        if method_found:
            # importante: acrescentamos o .bind(self) no retorno
            return method.bind(self)  
        else:
            raise AttributeError(attr)
```

Pronto! Nossos métodos estão associados a instâncias específicas e podemos usar
`this` impunemente nas nossas classes 🥳🎉