<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<style>
*{
  margin:0;
  padding: 0;
}
html,body{
  width:100%;
  height:100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  width: 100%;
  height:100%;
  box-sizing: border-box;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

/*滚动条里面小方块样式*/
::-webkit-scrollbar-thumb {
    border-radius: 100px;
    -webkit-box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
    background: #ccc;
}

/*滚动条里面轨道样式*/
::-webkit-scrollbar-track {
    -webkit-box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
    border-radius: 0;
    background: rgba(0, 0, 0, 0.1);
}
</style>
