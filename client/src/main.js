import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import * as d3 from 'd3';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import axios from './http';
import * as echarts from 'echarts';
Vue.prototype.$echarts = echarts



Vue.config.productionTip = false
Vue.use(ElementUI);
Vue.prototype.$d3 = d3;
Vue.prototype.$axios = axios;

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
