[medio]

Assim como fizemos com a implementação de funções, vamos dividir a implementação 
de classes em várias partes. Na primeira, e mais simples, vamos apenas dar o 
suporte sintático à declaração de classes. 

Para isso, implemente as regras de sintaxe da seção sobre [Class Declarations](https://craftinginterpreters.com/classes.html#class-declarations)

Assim como no caso das funções, vamos usar uma declaração genérica (e errada)
enquanto ainda não sabemos com implementar as classes corretamente. O método
eval da nossa classe `lox.ast.Class` deve portanto ficar como:

```python
class Class(Stmt):
    ...
    def eval(self, ctx: Ctx):
        # Crie uma classe LoxClass que por enquanto não faz nada. Salve-a no 
        # mesmo arquivo onde se encontra LoxFunction
        lox_class = LoxClass()  
        self.ctx[self.name] = LoxClass
```

As instâncias da classe `LoxClass` são responsáveis por representar as classes
Lox em tempo de execução. 