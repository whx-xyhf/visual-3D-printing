<template>
    <el-container>
    <el-header height="64px" style="box-shadow: 0px 2px 4px -1px rgb(0 0 0 / 20%), 0px 4px 5px 0px rgb(0 0 0 / 14%), 0px 1px 10px 0px rgb(0 0 0 / 12%);
    background-color: #f5f5f5;padding: 0">
    <div class="tool_bar_content">
        <div style="font-weight:700">
            Visual analysis of 3D printing results
        </div>
        <div class="spacer"></div>
        <div class="tool_bar__items">
            <el-button icon="el-icon-folder" style="height:56px;background:#f5f5f5;color:#000;border:none;transition: opacity .2s cubic-bezier(0.4, 0, 0.6, 1);"
            @click="clickFile">OPEN</el-button>
            <el-button icon="el-icon-download" style="height:56px;background:#f5f5f5;color:#000;border:none;transition: opacity .2s cubic-bezier(0.4, 0, 0.6, 1);"
            @click="handleDownload">SAVE</el-button>
            <input type="file" name="" id="fileInput" @change="uploadFile($event)" style="display:none">
        </div>
                    

    </div>
    </el-header>
    <el-container>
        <el-aside width="300px" style="height:calc(100vh - 64px);padding:0 4px">
            <el-tabs v-model="activeTabName">
                <el-tab-pane name="tool" style="padding:0 10px; overflow-y:auto">
                    <span slot="label"><i class="el-icon-s-tools"></i> TOOLS</span>
                    <div class="algorithm" :style="{'border': active_view_index == 0?'1px solid #5CB6FF':'1px solid #fff'}">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Image Contour Extraction &nbsp;</div> 
                        <!-- <div class=" runIcon el-icon-video-play" title="Run" @click="getSilhouette"></div> -->
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
                            <!-- <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Kernel Size:</div>
                                <el-slider v-model="kernel_size" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="10" :min="2" ></el-slider>
                                <div class="sliderValue" contenteditable="true" id="kernel_size">{{kernel_size}}</div>
                            </div> -->
                        </div>
                    </div>

                    <div class="algorithm" :style="{'border': active_view_index == 1?'1px solid #5CB6FF':'1px solid #fff'}">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Image Contour Fitting &nbsp;</div> 
                        <!-- <div class=" runIcon el-icon-video-play" title="Run" @click="getFinalContour"></div> -->
                        <div class="algorithm_content">
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Fitting Strength:</div>
                                <el-slider v-model="fitting_strength" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="20" :min="2" ></el-slider>
                            <div class="sliderValue" contenteditable="true" id="fitting_strength">{{fitting_strength}}</div>
                            </div>
                        </div>
                    </div>

                    <div class="algorithm" :style="{'border': active_view_index == 2?'1px solid #5CB6FF':'1px solid #fff'}">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Calculate Radius &nbsp;</div> 
                        <!-- <div class=" runIcon el-icon-video-play" title="Run" @click="getRadiusImg"></div> -->
                        <div class="algorithm_content">
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Radius Count:</div>
                                <el-slider v-model="radius_count" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="100" :min="2" ></el-slider>
                            <div class="sliderValue" contenteditable="true" id="radius_count">{{radius_count}}</div>
                            </div>
                        </div>
                    </div>
                    <div class="algorithm" :style="{'border': active_view_index == 2?'1px solid #5CB6FF':'1px solid #fff'}">
                        <div class="algorithm_title el-icon-arrow-down">&nbsp;&nbsp;Radius Data &nbsp;</div>
                        <!-- <div class="algorithm_content">
                            <div class="algorithm_content_sliderItem">
                                <div class="algorithm_content_demonstration">Radius Count:</div>
                                <el-slider v-model="radius_count" :show-tooltip="false" :style="{width:'40%',float:'left'}" :step="1" :max="100" :min="2" ></el-slider>
                            <div class="sliderValue" contenteditable="true" id="radius_count">{{radius_count}}</div>
                            </div>
                        </div> -->
                        <el-table
                            :data="tableData"
                            height="auto"
                            border
                            style="width: 100%">
                            <template slot="empty">
                                <el-empty :image-size="30" description='empty'></el-empty>
                            </template>
                            <el-table-column
                            prop="y"
                            label="Y"
                            width="60">
                            </el-table-column>
                            <el-table-column
                            prop="lr"
                            label="Left Radius"
                            width="70">
                            </el-table-column>
                            <el-table-column
                            prop="rr"
                            label="Right Radius"
                            width="70">
                            </el-table-column>
                            <el-table-column
                            prop="r"
                            label="Radius"
                            width="70">
                            </el-table-column>
                        </el-table>
                    </div>

                    
               
                </el-tab-pane>
                <el-tab-pane label="DATASETS" name="data" style="padding:0 10px; overflow-y:auto">
                    <span slot="label"><i class="el-icon-s-data"></i> DATASETS</span>
                </el-tab-pane>
            </el-tabs>
        </el-aside>
        <el-main style="height:calc(100vh - 64px); position:relative; width: calc(100% - 300px);padding:0;box-sizing:border-box;background: linear-gradient(rgb(51, 51, 51), rgb(153, 153, 153));" class="main">
            <div class="main_image_text triangle" :style="{'display':imageURl0.length>0?'block':'none'}"></div>
            
            <div class="main_img_box" style="width:100%;height:100%;z-index:1" :style="{'display':imageURl0.length>0?'block':'none'}" ref="imgbox0" @mouseover="isActive(0)" @mouseout="noActive">
                <div style="width:30px; height:30px;position:absolute;top:25px; right:25px;background:#202020; border-radius:10px">
                    <i :class="fullScreenFlag?'el-icon-remove-outline':'el-icon-full-screen'" @click="fullScreen(0)">
                    </i>
                </div>
                <img :src="imageURl0" alt="" srcset=""  id="img"  ref="img0"/>
                <div style="width:30px; height:30px;position:absolute;bottom:25px;left:50%;transform:translateX(-50%);background:#202020; border-radius:10px">
                    <!-- <i class="el-icon-edit svg_tool" style="left:5.5px;" @click="areaSelect(2)" :style="{'color': startPen2?'#5CB6FF':'#fff'}"></i>
                    <i class="el-icon-refresh svg_tool" style="left:30.5px;" @click="refresh(2)"></i> -->
                    <i class="el-icon-video-play svg_tool" style="left:5.5px;" @click="getSilhouette"></i>
                </div>
                <!-- <div class="main_image_text filename">Origin Picture</div> -->
            </div>

            <div class="main_img_box" style="width:100%;height:100%;z-index:2" :style="{'display':imageURl1.length>0?'block':'none'}" ref="imgbox1" @mouseover="isActive(1)" @mouseout="noActive">
                <img :src="imageURl1" alt="" srcset=""  id="img" ref="img1"/>
                <svg v-show="startPen1" ref="svg1" style="position:absolute; top: 16px; left:16px">
                    <!-- <rect :x="arae_select_rect1.x" :y="arae_select_rect1.y" :width="arae_select_rect1.width" :height="arae_select_rect1.height" fill="none" stroke="red" stroke-width="2px"></rect> -->
                    <circle v-for="(value, index) in points1" :key="index" :cx="value[0]" :cy="value[1]" r="3" fill="red"></circle>
                </svg>
                <div style="width:30px; height:30px;position:absolute;top:25px; right:25px;background:#202020; border-radius:10px">
                    <i :class="fullScreenFlag?'el-icon-remove-outline':'el-icon-full-screen'" @click="fullScreen(1)">
                    </i>
                </div>
                <div style="width:80px; height:30px;position:absolute;bottom:25px;left:50%;transform:translateX(-50%);background:#202020; border-radius:10px">
                    <i class="el-icon-edit svg_tool" style="left:5.5px;" @click="areaSelect(1)" :style="{'color': startPen1?'#5CB6FF':'#fff'}"></i>
                    <i class="el-icon-refresh svg_tool" style="left:30.5px;" @click="refresh(1)"></i>
                    <i class="el-icon-video-play svg_tool" style="left:55.5px;" @click="getFinalContour"></i>
                </div>
                <!-- <div class="main_image_text filename">Contour Extraction</div> -->
            </div>

            <div class="main_img_box" style="width:100%;height:100%;z-index:3" :style="{'display':imageURl2.length>0?'block':'none'}" ref="imgbox2"  @mouseover="isActive(2)" @mouseout="noActive">
                <img :src="imageURl2" alt="" srcset=""  id="img" ref="img2"/>
                <svg v-show="startPen2" ref="svg2" style="position:absolute; top: 16px; left:16px">
                    <!-- <rect :x="arae_select_rect1.x" :y="arae_select_rect1.y" :width="arae_select_rect1.width" :height="arae_select_rect1.height" fill="none" stroke="red" stroke-width="2px"></rect> -->
                    <circle v-for="(value, index) in points2" :key="index" :cx="value[0]" :cy="value[1]" r="3" fill="red"></circle>
                </svg>
                <div style="width:30px; height:30px;position:absolute;top:25px; right:25px;background:#202020; border-radius:10px">
                    <i :class="fullScreenFlag?'el-icon-remove-outline':'el-icon-full-screen'" @click="fullScreen(2)">
                    </i>
                </div>
                <div style="width:80px; height:30px;position:absolute;bottom:25px;left:50%;transform:translateX(-50%);background:#202020; border-radius:10px">
                    <i class="el-icon-edit svg_tool" style="left:5.5px;" @click="areaSelect(2)" :style="{'color': startPen2?'#5CB6FF':'#fff'}"></i>
                    <i class="el-icon-refresh svg_tool" style="left:30.5px;" @click="refresh(2)"></i>
                    <i class="el-icon-video-play svg_tool" style="left:55.5px;" @click="getRadiusImg"></i>
                </div>
                <!-- <div class="main_image_text filename">Image Contour Fitting</div> -->
            </div>

            <div class="main_img_box" style="width:100%;height:100%;z-index:4" :style="{'display':imageURl3.length>0?'block':'none'}" ref="imgbox3"  @mouseover="isActive(3)" @mouseout="noActive">
                <div style="width:30px; height:30px;position:absolute;top:25px; right:25px;background:#202020; border-radius:10px">
                    <i :class="fullScreenFlag?'el-icon-remove-outline':'el-icon-full-screen'" @click="fullScreen(3)"></i>
                </div>
                <img :src="imageURl3" alt="" srcset=""  id="img" ref="img3"/>
                <div style="width:30px; height:30px;position:absolute;bottom:25px;left:50%;transform:translateX(-50%);background:#202020; border-radius:10px">
                    <!-- <i class="el-icon-edit svg_tool" style="left:5.5px;" @click="areaSelect(2)" :style="{'color': startPen2?'#5CB6FF':'#fff'}"></i>
                    <i class="el-icon-refresh svg_tool" style="left:30.5px;" @click="refresh(2)"></i> -->
                    <i class="el-icon-video-play svg_tool" style="left:5.5px;" @click="getRadiusImg"></i>
                </div>
                <!-- <div class="main_image_text filename">Radius</div> -->
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
            imageURl0:'',
            imageURl1:'',
            imageURl2:'',
            imageURl3:'',
            activeTabName: 'tool',
            low_Threshold: 50,
            height_Threshold: 500,
            kernel_size:3,
            area_select:false,
            fitting_strength:8,
            points1:[],
            points2:[],
            fy1:'',
            fy2:'',
            fy3:'',
            radius_count:10,
            startPen1:false,
            startPen2:false,
            arae_select_rect1:{x:0,y:0,width:0,height:0},
            active_view_index:-1,
            fullScreenFlag:true,
            tableData:[]
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
                    if(this.fullScreenFlag){
                        this.fullScreenFlag = false;
                        this.fullScreen(0);
                    }
                    this.imageURl0 = fr.result;
                    
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
                    if(this.fullScreenFlag){
                        this.fullScreenFlag = false;
                        this.fullScreen(1);
                    }
                    this.imageURl1 = 'data:image/png;base64,' +  res.data.data;
                }
            })
        },
        // changeSwitch(value){
        //     if(value){
        //         this.points = [];
        //         const width = this.$refs.img1.clientWidth;
        //         const height = this.$refs.img1.clientHeight;
        //         this.$refs.svg1.style.width = width;
        //         this.$refs.svg1.style.height = height;
        //         this.$refs.svg1.onmousedown = (e)=>{
        //             this.points.push([e.offsetX, e.offsetY]);
        //         }
        //     }
        //     else{
        //         this.points = [];
        //         this.$refs.svg1.onmousedown = null;
        //     }
        // },
        getFinalContour(){
            const file = this.base64toFile(this.imageURl1, this.fileName);
            const formData = new FormData();
            const height = this.$refs.img1.clientHeight;
            const points = this.points1.map(v=>[v[0], height - v[1]])
            formData.append('file', file);
            formData.append('points', points);
            formData.append('width', this.$refs.img1.clientWidth);
            formData.append('height', this.$refs.img1.clientHeight);
            formData.append('fitting_strength', this.fitting_strength);
            let config = {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            }
            this.$axios.post('getFixContour', formData, config)
            .then(res=>{
                if(res.data.code == 200){
                    if(this.fullScreenFlag){
                        this.fullScreenFlag = false;
                        this.fullScreen(2);
                    }
                    this.imageURl2 = 'data:image/png;base64,' +  res.data.data.src;
                    // this.area_select = false;
                    this.fy1 = res.data.data.fy1;
                    this.fy2 = res.data.data.fy2;
                    this.fy3 = res.data.data.fy3;
                }
            })
        },
        getRadiusImg(){
            const height = this.active_view_index == 3? this.$refs.img3.clientHeight : this.$refs.img2.clientHeight;
            const points = this.points2.map(v=>[v[0], height - v[1]])
            const parameter = {
                fy1:this.fy1,
                fy2:this.fy2,
                fy3:this.fy3,
                points:points,
                img_ori_width: this.active_view_index == 3? this.$refs.img3.naturalWidth : this.$refs.img2.naturalWidth,
                img_ori_height: this.active_view_index == 3? this.$refs.img3.naturalHeight : this.$refs.img2.naturalHeight,
                show_width: this.active_view_index == 3? this.$refs.img3.clientWidth : this.$refs.img2.clientWidth,
                show_height: this.active_view_index == 3? this.$refs.img3.clientHeight : this.$refs.img2.clientHeight,
                count:this.radius_count
            }
            this.$axios.post('drawRadiusPic', parameter)
            .then(res=>{
                if(res.data.code == 200){
                    if(this.fullScreenFlag){
                        this.fullScreenFlag = false;
                        this.fullScreen(3);
                    }
                    this.imageURl3 = 'data:image/png;base64,' +  res.data.data.src;
                    console.log(res.data.data.r)
                    this.tableData = res.data.data.r.map((v, index)=>({
                        lr:Number(v[0]).toFixed(2),
                        rr:Number(v[1]).toFixed(2),
                        r:(Number(v[0]) + Number(v[1])).toFixed(2),
                        y:Number(res.data.data.y[index]).toFixed(2)
                    }))
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
        areaSelect(index){
            if(index==1){
                this.startPen1 = !this.startPen1;
                if(this.startPen1){
                    this.$refs.svg1.style.width = this.$refs.img1.clientWidth;
                    this.$refs.svg1.style.height = this.$refs.img1.clientHeight;
                    this.$refs.svg2.style.width = this.$refs.img2.clientWidth;
                    this.$refs.svg2.style.height = this.$refs.img2.clientHeight;
                    this.$refs.svg1.onmousedown = (e)=>{
                        if(this.points1.length == 3) {
                            this.points1 = [];
                            this.points2 = [];
                        }
                        this.points1.push([e.offsetX, e.offsetY]);
                        this.points2.push([e.offsetX, e.offsetY]);
                    }
                }
                
            }else{
                this.startPen2 = !this.startPen2;
                if(this.startPen2){
                    this.$refs.svg2.style.width = this.$refs.img2.clientWidth;
                    this.$refs.svg2.style.height = this.$refs.img2.clientHeight;
                    this.$refs.svg2.onmousedown = (e)=>{
                        if(this.points2.length == 3) this.points2 = [];
                        // this.points1.push([e.offsetX, e.offsetY]);
                        this.points2.push([e.offsetX, e.offsetY]);
                    }
                }
            }
        },
        // areaSelect(index){
        //     if(index==1){
        //         this.startPen1 = !this.startPen1;
        //         if(this.startPen1){
        //             this.$refs.svg1.style.width = this.$refs.img1.clientWidth;
        //             this.$refs.svg1.style.height = this.$refs.img1.clientHeight;
        //             this.$refs.svg2.style.width = this.$refs.img2.clientWidth;
        //             this.$refs.svg2.style.height = this.$refs.img2.clientHeight;
        //             this.$refs.svg1.onmousedown = (e)=>{
        //                 this.arae_select_rect1.width = 0;
        //                 this.arae_select_rect1.height = 0;
        //                 this.arae_select_rect1.x = e.offsetX;
        //                 this.arae_select_rect1.y = e.offsetY;
        //                 this.$refs.svg1.onmousemove = (e1)=>{
        //                     let width = e1.offsetX - e.offsetX;
        //                     let height = e1.offsetY - e.offsetY;
        //                     if(width>0 && height>0){
        //                         this.arae_select_rect1.width = width;
        //                         this.arae_select_rect1.height = height;
        //                     }
        //                 }
        //                 this.$refs.svg1.onmouseup = (ee)=>{
        //                     this.$refs.svg1.onmousemove = null;
        //                     this.points1 = [[e.offsetX, e.offsetY], [e.offsetX, ee.offsetY], [ee.offsetX, ee.offsetY]];
        //                     this.points2 = [[e.offsetX, e.offsetY], [e.offsetX, ee.offsetY], [ee.offsetX, ee.offsetY]];
        //                 }
        //             }
        //         }
                
        //     }else{
        //         this.startPen2 = !this.startPen2;
        //         if(this.startPen2){
        //             this.$refs.svg2.style.width = this.$refs.img2.clientWidth;
        //             this.$refs.svg2.style.height = this.$refs.img2.clientHeight;
        //             this.$refs.svg2.onmousedown = (e)=>{
        //                 this.arae_select_rect1.width = 0;
        //                 this.arae_select_rect1.height = 0;
        //                 this.arae_select_rect1.x = e.offsetX;
        //                 this.arae_select_rect1.y = e.offsetY;
        //                 this.$refs.svg2.onmousemove = (e1)=>{
        //                     let width = e1.offsetX - e.offsetX;
        //                     let height = e1.offsetY - e.offsetY;
        //                     if(width>0 && height>0){
        //                         this.arae_select_rect1.width = width;
        //                         this.arae_select_rect1.height = height;
        //                     }
        //                 }
        //                 this.$refs.svg2.onmouseup = (ee)=>{
        //                     this.$refs.svg2.onmousemove = null;
        //                     this.points1 = [[e.offsetX, e.offsetY], [e.offsetX, ee.offsetY], [ee.offsetX, ee.offsetY]];
        //                     this.points2 = [[e.offsetX, e.offsetY], [e.offsetX, ee.offsetY], [ee.offsetX, ee.offsetY]];
        //                 }
        //             }
        //         }
        //     }
        // },
        refresh(index){
            if(index==1){
                this.arae_select_rect1 = {x:0,y:0,width:0,height:0};
                this.points1 = [];
            }
            else{
                this.arae_select_rect1 = {x:0,y:0,width:0,height:0};
                this.points2 = [];
            }
        },
        fullScreen(index){
            this.fullScreenFlag = !this.fullScreenFlag;
            if(this.fullScreenFlag){
                if(index == 0){
                    this.$refs.imgbox0.style.top = 0;
                    this.$refs.imgbox0.style.left = 0;
                    this.$refs.imgbox0.style.width = '100%';
                    this.$refs.imgbox0.style.height = '100%';
                    this.$refs.imgbox0.style.display = 'block';
                    this.$refs.imgbox1.style.display = 'none';
                    this.$refs.imgbox2.style.display = 'none';
                    this.$refs.imgbox3.style.display = 'none';
                    // this.$refs.imgbox0.style.zIndex = 5;
                }
                else if(index == 1){
                    this.$refs.imgbox1.style.top = 0;
                    this.$refs.imgbox1.style.left = 0;
                    this.$refs.imgbox1.style.width = '100%';
                    this.$refs.imgbox1.style.height = '100%';
                    this.$refs.imgbox0.style.display = 'none';
                    this.$refs.imgbox1.style.display = 'block';
                    this.$refs.imgbox2.style.display = 'none';
                    this.$refs.imgbox3.style.display = 'none';
                    // this.$refs.imgbox1.style.zIndex = 5;
                }
                else if(index == 2){
                    this.$refs.imgbox2.style.top = 0;
                    this.$refs.imgbox2.style.left = 0;
                    this.$refs.imgbox2.style.width = '100%';
                    this.$refs.imgbox2.style.height = '100%';
                    this.$refs.imgbox0.style.display = 'none';
                    this.$refs.imgbox1.style.display = 'none';
                    this.$refs.imgbox2.style.display = 'block';
                    this.$refs.imgbox3.style.display = 'none';
                    // this.$refs.imgbox2.style.zIndex = 5;
                }
                else if(index == 3){
                    this.$refs.imgbox3.style.top = 0;
                    this.$refs.imgbox3.style.left = 0;
                    this.$refs.imgbox3.style.width = '100%';
                    this.$refs.imgbox3.style.height = '100%';
                    this.$refs.imgbox0.style.display = 'none';
                    this.$refs.imgbox1.style.display = 'none';
                    this.$refs.imgbox2.style.display = 'none';
                    this.$refs.imgbox3.style.display = 'block';
                    // this.$refs.imgbox3.style.zIndex = 5;
                    
                }
                this.active_view_index = index;
            }
            else{
                this.$refs.imgbox0.style.zIndex = 1;
                this.$refs.imgbox1.style.zIndex = 2;
                this.$refs.imgbox2.style.zIndex = 3;
                this.$refs.imgbox3.style.zIndex = 4;
                this.$refs.imgbox0.style.width = '50%';
                this.$refs.imgbox0.style.height = '50%';
                this.$refs.imgbox0.style.top = 0;
                this.$refs.imgbox0.style.left = 0;
                this.$refs.imgbox1.style.width = '50%';
                this.$refs.imgbox1.style.height = '50%';
                this.$refs.imgbox1.style.top = 0;
                this.$refs.imgbox1.style.left = '50%';
                this.$refs.imgbox2.style.width = '50%';
                this.$refs.imgbox2.style.height = '50%';
                this.$refs.imgbox2.style.top = '50%';
                this.$refs.imgbox2.style.left = 0;
                this.$refs.imgbox3.style.width = '50%';
                this.$refs.imgbox3.style.height = '50%';
                this.$refs.imgbox3.style.top = '50%';
                this.$refs.imgbox3.style.left = '50%';
                this.$refs.imgbox0.style.display = 'block';
                if(this.imageURl1.length>0) this.$refs.imgbox1.style.display = 'block';
                if(this.imageURl2.length>0) this.$refs.imgbox2.style.display = 'block';
                if(this.imageURl3.length>0) this.$refs.imgbox3.style.display = 'block';
                this.startPen1 = false;
                this.startPen2 = false;
                this.active_view_index = -1;
            }
            
        },
        isActive(index){
            this.active_view_index = index;
        },
        noActive(){
            this.active_view_index = -1;
        },
        handleDownload() {
            if(this.tableData.length > 0){
                import('@/vendor/Export2Excel').then(excel => {
                    const tHeader = ['Y', 'Left Radius', 'Right Radius', 'Radius'];
                    const filterVal = ['y', 'lr', 'rr', 'r'];
                    
                    const data = this.formatJson(filterVal)
                    excel.export_json_to_excel({
                    header: tHeader,
                    data,
                    filename: this.fileName + '_Radius'
                    })
                })
            }
            else{
                alert('The data is Empty');
            }
        },
        formatJson(filterVal) {
        return this.tableData.map(v => filterVal.map(j => {
            return v[j]
        }))
        },
        // getContainedSize(img) {
        //     var ratio = img.naturalWidth/img.naturalHeight
        //     var width = img.height*ratio
        //     var height = img.height
        //     console.log(img.naturalWidth, img.naturalHeight, img.width, img.height)
        //     if (width > img.width) {
        //         width = img.width
        //         height = img.width/ratio
        //     }
        //     return [width, height]
        // }
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
    position:absolute;
    transition: all .2s cubic-bezier(0.4, 0, 0.6, 1);
    padding:20px;
    top:0;
    left:0;
    /* float:left; */

}
.main_img_box img{
    object-fit:contain;
    width:100%;
    height:100%;
    user-select: none;
}
.main_image_text{
    color:white;
    position: absolute;
}
.filename{
    top:20px;
    left:40px;
    color: #000;
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
    box-sizing: border-box;
    /* border-top:1px solid #ccc; */
    border-bottom:1px solid #ccc !important;
}
.algorithm_content_sliderItem{
    /* padding:0 0 0 20px; */
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
.el-icon-full-screen{
    font-size:20px; 
    color:white;
    position:absolute;
    top:5.5px;
    left:5.5px;
    cursor:pointer;
}
.el-icon-full-screen:hover{
    color:#5CB6FF;
}
.el-icon-remove-outline{
    font-size:20px; 
    color:white;
    position:absolute;
    top:5.5px;
    left:5.5px;
    cursor:pointer;
}
.el-icon-remove-outline:hover{
    color:#5CB6FF;
}
.svg_tool{
    font-size:20px; 
    color:white;
    cursor:pointer;
    position:absolute;
    top:5.5px;
}
.svg_tool:hover{
    color:#5CB6FF;
}
</style>
<style>
.el-slider__button{
    border-radius: 30% !important;
}
</style>