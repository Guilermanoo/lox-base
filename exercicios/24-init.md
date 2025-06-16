[medio]

O capítulo
[Constructors and Initializers](https://craftinginterpreters.com/classes.html#constructors-and-initializers)
define como os construtures, ou para ser mais preciso, inicializadores,
funcionam em Lox.

Em Lox os inicializadores são declarados pelo método init() no corpo de uma
classe.

```lox
class Pt {
    init(x, y) {
        this.x = x;
        this.y = y;
    }
}
```

Ao construir instâncias novas, devemos portanto executar o método init
automaticamente se ele estiver definido na classe ou em uma superclasse. Isso
pode ser feito modificando o método `LoxClass.__call__`

```python
class LoxClass:
    ...

    def __call__(self, *args):
        instance = LoxInstance()
        ...
        if has_init:
            bound_init = init.bind(instance)
            bound_init(*args)
        return instance
```

Apesar de parecer com uma declaração de método normal, o init se comporta de
forma diferente das outras funções. Ele é executado automaticamente quando as
instâncias são criadas e se for chamado novamente com um método de uma
instância, ele deve ser executado e retornar this ao final.

```lox
var pt = Pt(1, 2);
var new = pt.init(3, 4);  // pt e new são o mesmo objeto
print new.x == pt.x and pt.x == 3; // true
```
 
Isso é anormal, porque o corpo da execução do método init nunca retorna `this`
explicitamente. Nenhum outro método tem esse comportamento. Podemos suportar o 
comportamento anormal do método init criando um método que lida com esse caso no
`LoxInstance`:

```python
class LoxInstance:
    ...
    def init(self, *args):
        init = ... # pega "init" na classe da instância
        bound_init = init.bind(self)
        bound_init(*args)
        return self  # retornamos a instância e não o resultado de init
``` 

Complete a implementação dessas duas classes para passar em todos os testes.