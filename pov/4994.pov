#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 1 }        
    sphere {  m*<0.5328852449344144,0.2896916311538995,8.257476108420953>, 1 }
    sphere {  m*<2.4679518660236868,-0.03580636981537289,-2.895394515102316>, 1 }
    sphere {  m*<-1.8883718878754603,2.190633599216852,-2.6401307550671023>, 1}
    sphere { m*<-1.6205846668376285,-2.697058343187045,-2.4505844699045296>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5328852449344144,0.2896916311538995,8.257476108420953>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5 }
    cylinder { m*<2.4679518660236868,-0.03580636981537289,-2.895394515102316>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5}
    cylinder { m*<-1.8883718878754603,2.190633599216852,-2.6401307550671023>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5 }
    cylinder {  m*<-1.6205846668376285,-2.697058343187045,-2.4505844699045296>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 1 }        
    sphere {  m*<0.5328852449344144,0.2896916311538995,8.257476108420953>, 1 }
    sphere {  m*<2.4679518660236868,-0.03580636981537289,-2.895394515102316>, 1 }
    sphere {  m*<-1.8883718878754603,2.190633599216852,-2.6401307550671023>, 1}
    sphere { m*<-1.6205846668376285,-2.697058343187045,-2.4505844699045296>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5328852449344144,0.2896916311538995,8.257476108420953>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5 }
    cylinder { m*<2.4679518660236868,-0.03580636981537289,-2.895394515102316>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5}
    cylinder { m*<-1.8883718878754603,2.190633599216852,-2.6401307550671023>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5 }
    cylinder {  m*<-1.6205846668376285,-2.697058343187045,-2.4505844699045296>, <-0.2667565279825703,-0.13784034520174704,-1.666184989651134>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    