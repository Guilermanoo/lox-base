[médio]

Na atividade anterior, vimos que as instâncias de `LoxClass` são responsáveis por
criar objetos Lox (representadas como instâncias de `LoxInstance` em Python).
O objetivo dessa atividade é completar a implementação de `LoxInstance`.

Os objetos do tipo `LoxInstance` são responsáveis por algumas funcionalidades:

1. Guardar os atributos do objeto definidos em Lox, ex: `lox.attr = 42;`.
2. Encontrar os métodos associados ao objeto, ex: `lox.method(42);`.

A primeira parte é trivial: o Python já possui a mesma semântica de atributos
que em Lox. Ou seja, tanto em Python quanto em Lox, podemos criar e redefinir
atributos mais ou menos a vontade e para implementar a funcionalidade (1) na
nossa `LoxInstance`, podemos delegar para o Python e não precisamos fazer nada!
Por exemplo, o código abaixo funciona tanto em Python quanto em Lox:

```python
obj.foo = "foo";
obj.bar = "bar";
print(obj.foo);
print(obj.baz);
```

Aqui assumimos que obj foi criado instanciando uma classe definida pelo usuário.
Coloquei os `;` para agradar o Lox e os parênteses em `print` para agradar o
Python. O código acima funciona em ambas as linguagens. E nos dois casos, a 
última linha produzirá um erro ou um valor dependendo se `obj.baz` foi definido
previamente ou não.

A segunda parte de implementar suporte a métodos é um pouco mais complicada. Os
métodos são definidos na classe e não na instância. Deste modo, nossa
`LoxInstance` deve guardar uma referência à classe associada àquela instâcia.
Recomendo salvar essa referência em um atributo privado, para evitar colisões de
nomes com os atributos definidos em Lox. Em Python, os atributos privados
possuem nomes começam com dois underscores. Eles também não são 100% privados,
mas vamos fingir que são ;)

Modifique LoxInstance para salvar a classe como atributo da instância e mude a
implementação de `LoxClass.__call__` para passar a classe explicitamente para o
construtor de `LoxInstance`. Com isso, nossas instâncias agora sabem a qual
classe elas pertencem.

Por enquanto, se tentarmos acessar um método em uma `LoxInstance`, o
interpretador terminará com um AttributeError já que tentamos acessar o método
como atributo e ele não existe no objeto e sim na classe. Python permite
declarar um método que é executado no acesso a um atributo que não existe.
Usaremos isso para buscar um método válido na classe e retorná-lo, caso
encontre:

```python
class LoxInstance:
    ...
    def __getattr__(self, attr):
        """
        self.__getattr__(self, "attr") <==> self.attr 
            (se o objeto não definir attr explicitamente)
        """
        # Aqui attr é possivelmente o nome de um método, procure-o na classe!
        ...
        if method_found:
            return method
        else:
            raise AttributeError(attr)
```

Os testes nessa atividade verificam se você implementou corretamente essas
funcionalidades básicas em `LoxClass` e no `LoxInstance`