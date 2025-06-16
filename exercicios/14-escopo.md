[dificil]

Existe, até agora, um problema sério na nossa implementação do escopo de
variáveis do nosso interpretador. Usamos um dicionário que grava localmente a
associação entre nomes de variáveis e seus valores. Isso significa que no nosso
interpretador todas as variáveis são globais!

É lógico que precisamos mudar isso. O grande culpado aqui é a nossa classe de
contexto. Por enquanto `lox.ctx.Ctx` é apenas uma sub-classe de dicionários e
todas as variáveis são armazenadas juntas, não importa o bloco em que foram
definidas. Vamos reimplementar nosso contexto como uma lista encadeada de
dicionários. Dessa forma, cada vez que entramos num bloco, colocamos um
dicionário no topo da pilha e ao sair, removemos o mesmo limpando todas as
variáveis declaradas no mesmo.

O código abaixo ilustra a estratégia:

```lox
// Começamos com ctx = [{}]
var x = 1;  // agora temos ctx = [{"x": 1}]
{
    // ao abrir o bloco, colocamos um novo dicionário vazio no topo da pilha.
    // agora temos: ctx = [{"x": 1}, {}]
    var y = 2;  // ctx = [{"x": 1}, {"y": 2}]
    var x = 3;  // ctx = [{"x": 1}, {"y": 2, "x": 3}]

    // acessamos a pilha de cima para baixo durante a resolução de variáveis.
    // assim, um acesso a x nesse escopo usaria x definido no último dicionário
    print x;  // imprime 3
}
// Ao sair do escopo, removemos o dicionário no topo da pilha. Isso limpa todas
// as variáveis definidas nesse bloco
// ctx = [{"x": 1}]
print x;  // agora imprime 1
```

É lógico que não podemos simplesmente trocar nossos dicionários pela nova
estrutura de dados: isso faria todo código antigo que espera que o contexto seja
um dicionário quebrar. Vamos implementar uma interface que faça ela se comportar
mais ou menos como um dicionário implementando alguns métodos estratégicos.

```
from typing import Optional
from dataclasses import dataclass

@dataclass
class Ctx:
    scope: dict
    parent: Optional["Ctx"] 

    def __getitem__(self, key: str) -> Value:
        "self.__getitem__(key) <==> self[key]"

    def __setitem__(self, key: str, value: Value):
        "self.__setitem__(key, value) <==> self[key] = value"
```

Os métodos `__getitem__` e `__setitem__` são executados por baixo dos panos
sempre que indexamos um objeto no Python. Desse modo, o interpretador transforma
todo acesso do tipo `ctx[key]` em `ctx.__getitem__(key)`. De forma semelhante,
`ctx[key] = value` é traduzido em `ctx.__setitem__(key, value)`.

Implemente essas duas funções usando a seguinte lógica:

* ao ler uma variável, buscamos no escopo atual. Se na variável não existir, 
  buscamos recursivamente no contexto pai até chegar à base. Se a variável não
  for encontrada, levante KeyError.
* ao escrever uma variável, buscamos no escopo atual. Se existir, sobrescrevemos
  com o novo valor. Se na variável não existir, buscamos recursivamente no
  contexto pai até chegar à base. Se a variável não for encontrada, levante
  KeyError.

Também é necessário implementar um novo método que crie uma variável nova no
escopo atual, independentemente dela exisitr ou não em algum escopo pai. Vamos
chamar esse método de var_def. Assim, precisamos fazer a seguinte tradução:

    Lox                Python
    x = value      =>  ctx["x"] = value
    var x = value  =>  ctx.var_def("x", value)

Adapte os métodos Assign.eval e VarDef.eval para utilizar o método correto.

Durante essa atividade, será necessário implementar alguns outros métodos da 
classe `Ctx`. Esse processo é guiado pelas mensagens de erro nos testes.

No fim da atividade, teremos uma classe Ctx capaz de fazer tudo que a antiga
implementação fazia e algumas coisas a mais. Podemos esperar que todos os testes
antigos passem ao final da implementação.
