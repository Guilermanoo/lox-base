[difícil]

Nosso interpretador aceita vários exemplos de código demonstravelmente inválidos
que irão causar problemas em tempo de execução.

Algumas verificações simples permitem detectar alguns desses erros, o que 
podemos pensar como uma forma rudimentar de análise semântica. Nossas classes no
módulo AST já possuem um mecanismo pronto para realizar este tipo de análise, 
mas ainda não o utilizamos até o momento.

Os dois métodos importantes são `Node.validate_self(cursor)` e
`Node.validate_tree()`. O primeiro valida um nó específico da árvore sintática e
podemos sobrescrevê-lo para implementar validações para cada um dos nós criados
no módulo `lox.ast`. Já o segundo percorre a árvore inteira e executa
`validate_self` em cada nó.

Nesse exercício, vamos implementar `validate_self()` em algumas classes para
detectar erros bobos. Por exemplo, podemos sobrescrever este método na classe
`lox.ast.VarDef` para previnir nomes de variáveis inválidos:

```python
# Importe essas duas classes no topo do módulo!
from .node import Cursor
from .errors import SemanticError


@dataclass
class VarDef(Expr):
    ...

    def validate_self(self, cursor: Cursor):
        ... # analiza se existe erro

        # Levantamos uma exceção em caso de erro e não fazemos nada, caso 
        # contrário
        if error:
            # A exceção correta vai depender do erro, mas nesse exercício 
            # usaremos lox.errors.SemanticError, passando explicitamente o 
            # nome do token inválido como argumento.
            raise SemanticError("nome inválido", token=self.name) 
```

Você deve fazer as verificações abaixo. Cada verificação pode estar associada a
uma classe diferente, então escolha de forma apropriada.

* Variáveis não podem se chamar true, false, nil ou qualquer outra palavra      
  reservada
* Blocos não podem ter duas declarações de variáveis com nomes idênticos.
* Funções não podem ter dois parâmetros com o mesmo nome.
* Uma função não pode declarar variáveis em seu corpo com o mesmo nome que um 
  argumento.

Em todos esses exemplos, pode ser útil utilizar os conjuntos no Python. Eles são
coleções lineares como listas, mas eliminam duplicações. Assim, para verificar
se existem elementos duplicados em uma lista, basta convertê-la para um conjunto
e testar se o tamanho se modificou.