<template>
    <el-container>
    <el-header height="64px" style="box-shadow: 0px 2px 4px -1px rgb(0 0 0 / 20%), 0px 4px 5px 0px rgb(0 0 0 / 14%), 0px 1px 10px 0px rgb(0 0 0 / 12%);
    background-color: #f5f5f5;">
    <div class="tool_bar_content">
        <div style="font-weight:700">
            Visual analysis of 3D printing results
        </div>
        <div class="spacer"></div>
        <div class="tool_bar__items">
            <el-button icon="el-icon-folder" style="height:56px;background:#f5f5f5;color:#000;border:none;transition: opacity .2s cubic-bezier(0.4, 0, 0.6, 1);"
            @click="clickFile">OPEN</el-button>
            <el-button icon="el-icon-download" style="height:56px;background:#f5f5f5;color:#000;border:none;transition: opacity .2s cubic-bezier(0.4, 0, 0.6, 1);">SAVE</el-button>
            <input type="file" name="" id="fileInput" @change="uploadFile($event)" style="display:none">
        </div>
                    

    </div>
    </el-header>
    <el-container>
        <el-aside width="300px" style="height:calc(100vh - 64px);padding:0 4px">
            <el-tabs v-model="activeTabName">
                <el-tab-pane name="tool" style="padding:0 10px">
                    <span slot="label"><i class="el-icon-s-tools"></i> TOOLS</span>
                    <div class="algorithm">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Image Contour Extraction &nbsp;</div> <div class=" runIcon el-icon-video-play" title="Run" @click="getSilhouette"></div>
                        <!-- <div class="algorithm_title">Image Contour Extraction</div> -->
                        <div class="algorithm_content">
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Min Threshold:</div>
                                <el-slider v-model="low_Threshold" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="200" :min="10" ></el-slider>
                                <div class="sliderValue" contenteditable="true" id="low_Threshold">{{low_Threshold}}</div>
                            </div>
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Max Threshold:</div>
                                <el-slider v-model="height_Threshold" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="1000" :min="200" ></el-slider>
                                <div class="sliderValue" contenteditable="true" id="height_Threshold">{{height_Threshold}}</div>
                            </div>
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Kernel Size:</div>
                                <el-slider v-model="kernel_size" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="10" :min="2" ></el-slider>
                                <div class="sliderValue" contenteditable="true" id="kernel_size">{{kernel_size}}</div>
                            </div>
                        </div>
                    </div>
                    <div class="algorithm">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Image Contour Fitting &nbsp;</div> <div class=" runIcon el-icon-video-play" title="Run" @click="getFinalContour"></div>
                        <div class="algorithm_content">
                            <div class="algorithm_content_sliderItem" style="height:40px">
                                <div class="algorithm_content_demonstration">Area Select:</div>
                                <el-switch
                                    v-model="area_select"
                                    active-color="#13ce66"
                                    inactive-color="#ff4949"
                                    style="top:8px"
                                    @change="changeSwitch"
                                    >
                                </el-switch>
                            </div>
                        </div>
                    </div>
                    <!-- <button @click="getSilhouette">获取轮廓</button>
                    <button>轮廓拟合</button> -->
                </el-tab-pane>
                <el-tab-pane label="DATASETS" name="data" style="padding:0 10px">
                    <span slot="label"><i class="el-icon-s-data"></i> DATASETS</span>
                </el-tab-pane>
            </el-tabs>
        </el-aside>
        <el-main style="height:calc(100vh - 64px); position:relative; width: calc(100% - 300px);padding:0;box-sizing:border-box" class="main">
            <div class="main_img_box" style="width:100%;height:100%">
                <div class="main_image_text triangle" :style="{'display':imageURl.length>0?'block':'none'}"></div>
                <div class="main_image_text filename">{{ fileName }}</div>
                <img :src="imageURl" alt="" srcset=""  id="img" :style="{'display':imageURl.length>0?'block':'none'}" ref="img"/>
                <svg v-show="changeSwitch" ref="svg" style="position:absolute; top: 16px; left:16px">
                    <circle v-for="(value, index) in points" :key="index" :cx="value[0]" :cy="value[1]" r="3" fill="blue"></circle>
                </svg>
            </div>
            
        </el-main>
    </el-container>
    </el-container>
</template>

<script>
export default {
    name:'Layout',
    data(){
        return{
            fileName:'',
            imageURl:'',
            activeTabName: 'tool',
            low_Threshold: 50,
            height_Threshold: 500,
            kernel_size:3,
            area_select:false,
            points:[],
        }
    },
    methods:{
        clickFile(){
            document.getElementById("fileInput").click();
        },
        uploadFile(e){
            const fileName = e.target.files[0].name.split('.')[0];
            this.fileName = fileName;
            // 
            // this.$store.dispatch('updateFileName',fileName);
            // const file = e.target.files[0];
            // const formData = new FormData();
            // formData.append('file', file);
            // let config = {
            //     headers: {
            //         'Content-Type': 'multipart/form-data'
            //     }
            // }
            // this.$axios.post('upload_file', formData, config)
            // .then(res=>{
            //     this.$store.dispatch('updateOriginData', JSON.parse(res.data.data));
            //     this.$store.dispatch('updateSamplingData', []);
            //     this.$store.dispatch('updateLabels', []);
            //     this.sampling_rate = 100;
            // })

            if(window.FileReader){
                //创建读取文件的对象
                var fr = new FileReader();
                //以读取文件字符串的方式读取文件 但是不能直接读取file 
                //因为文件的内容是存在file对象下面的files数组中的
                //该方法结束后图片会以data:URL格式的字符串（base64编码）存储在fr对象的result中
                fr.readAsDataURL(e.target.files[0]);
                fr.onloadend = ()=>{
                    this.imageURl = fr.result;
                    
                }
            }
        },
        getSilhouette(){
            const file = document.getElementById('fileInput').files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('low_Threshold', this.low_Threshold);
            formData.append('height_Threshold', this.height_Threshold);
            formData.append('kernel_size', this.kernel_size)
            let config = {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            }
            this.$axios.post('getSilhouette', formData, config)
            .then(res=>{
                if(res.data.code == 200){
                    this.imageURl = 'data:image/png;base64,' +  res.data.data;
                }
            })
        },
        changeSwitch(value){
            if(value){
                this.points = [];
                const width = this.$refs.img.clientWidth;
                const height = this.$refs.img.clientHeight;
                this.$refs.svg.style.width = width;
                this.$refs.svg.style.height = height;
                this.$refs.svg.onmousedown = (e)=>{
                    this.points.push([e.offsetX, e.offsetY]);
                }
            }
            else{
                this.points = [];
                this.$refs.svg.onmousedown = null;
            }
        },
        getFinalContour(){
            const file = this.base64toFile(this.imageURl, this.fileName);
            const formData = new FormData();
            formData.append('file', file);
            formData.append('points', this.points);
            formData.append('width', this.$refs.img.clientWidth);
            formData.append('height', this.$refs.img.clientHeight);
            let config = {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            }
            this.$axios.post('getFixContour', formData, config)
            .then(res=>{
                if(res.data.code == 200){
                    this.imageURl = 'data:image/png;base64,' +  res.data.data.src;
                    this.area_select = false;
                }
            })
        },
        base64toFile(data, fileName) {
            const dataArr = data.split(",");
            const byteString = atob(dataArr[1]);
            const options = {
                type: "image/png",
                endings: "native"
            };
            const u8Arr = new Uint8Array(byteString.length);
            for (let i = 0; i < byteString.length; i++) {
                u8Arr[i] = byteString.charCodeAt(i);
            }
            return new File([u8Arr], fileName + ".jpg", options);//返回文件流
        },
    }
}
</script>
<style scoped>
.tool_bar_content{
    height: 64px;
    padding: 4px 16px;
    display: flex;
    position: relative;
    align-items: center
}
.tool_bar__items button:hover{
    background-color: #ccc;
}

.tool_bar_btn__content{
    align-items: center;
    color: inherit;
    display: flex;
    flex: 1 0 auto;
    justify-content: inherit;
    line-height: normal;
    position: relative;
    transition: inherit;
    transition-property: opacity;
}
.spacer{
    flex-grow: 1 !important;
}
.tool_bar__items{
    display: flex;
    height: inherit;
}
.main_img_box {
    padding: 16px;
    box-sizing: border-box;
    background: linear-gradient(rgb(51, 51, 51), rgb(153, 153, 153));
    position:relative;
    transition: all .2s cubic-bezier(0.4, 0, 0.6, 1);
}
.main_img_box img{
    object-fit:contain;
    width:100%;
    height:100%
}
.main_image_text{
    color:white;
    position: absolute;
}
.filename{
    top:8px;
    left:8px;
}
.triangle{
    left:1px;
    top:1px;
    width:0;
    height:0;
    border-bottom:8px solid transparent;
    border-left:8px solid #fff;
    border-right:8px solid transparent;
}
.algorithm{
    text-align: left;
    float:left;
    height:auto;
    width:100%;
    margin-bottom:10px;
    /* border-top:1px solid #ccc; */
    border-bottom:1px solid #ccc;
}
.algorithm_content_sliderItem{
    padding:0 0 0 20px;
    float: left;
    width: 100%;
}
.algorithm_content_demonstration{
    font-size: 12px;
    float:left;
    width:36%;
    position: relative;
    top:12px;
    text-align: left;
    font-weight: 400;
}
.sliderValue{
    float:left;
    margin: 12px 12px;
    font-size:10px;
    text-align: center;
    width:calc(25% - 32px);
    border:none;
    border-bottom:1px solid #ccc;
}
.algorithm_title{
    margin-bottom:10px;
    font-weight: 500;
    font-size: 0.875rem;
}
</style>
<style>
.el-slider__button{
    border-radius: 30% !important;
}
</style>