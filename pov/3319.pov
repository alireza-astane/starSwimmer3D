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
    sphere { m*<0.2748757958405948,0.727491685570176,0.031210844341180738>, 1 }        
    sphere {  m*<0.5156109005822864,0.8562017637505015,3.018765615461731>, 1 }
    sphere {  m*<3.0095841898468514,0.8295256609565503,-1.1979986811100027>, 1 }
    sphere {  m*<-1.346739564052296,3.055965629988778,-0.9427349210747885>, 1}
    sphere { m*<-3.227026323331155,-5.892352202551786,-1.9977691445809311>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5156109005822864,0.8562017637505015,3.018765615461731>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5 }
    cylinder { m*<3.0095841898468514,0.8295256609565503,-1.1979986811100027>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5}
    cylinder { m*<-1.346739564052296,3.055965629988778,-0.9427349210747885>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5 }
    cylinder {  m*<-3.227026323331155,-5.892352202551786,-1.9977691445809311>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5}

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
    sphere { m*<0.2748757958405948,0.727491685570176,0.031210844341180738>, 1 }        
    sphere {  m*<0.5156109005822864,0.8562017637505015,3.018765615461731>, 1 }
    sphere {  m*<3.0095841898468514,0.8295256609565503,-1.1979986811100027>, 1 }
    sphere {  m*<-1.346739564052296,3.055965629988778,-0.9427349210747885>, 1}
    sphere { m*<-3.227026323331155,-5.892352202551786,-1.9977691445809311>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5156109005822864,0.8562017637505015,3.018765615461731>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5 }
    cylinder { m*<3.0095841898468514,0.8295256609565503,-1.1979986811100027>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5}
    cylinder { m*<-1.346739564052296,3.055965629988778,-0.9427349210747885>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5 }
    cylinder {  m*<-3.227026323331155,-5.892352202551786,-1.9977691445809311>, <0.2748757958405948,0.727491685570176,0.031210844341180738>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    