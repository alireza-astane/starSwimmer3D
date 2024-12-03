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
    sphere { m*<1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 1 }        
    sphere {  m*<1.3833405699944243,1.2743903680116328e-18,3.679662361473433>, 1 }
    sphere {  m*<4.5730155958237475,7.144676479647779e-18,-0.7980479722731812>, 1 }
    sphere {  m*<-3.771283401883407,8.164965809277259,-2.300600867329175>, 1}
    sphere { m*<-3.771283401883407,-8.164965809277259,-2.3006008673291785>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3833405699944243,1.2743903680116328e-18,3.679662361473433>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5 }
    cylinder { m*<4.5730155958237475,7.144676479647779e-18,-0.7980479722731812>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5}
    cylinder { m*<-3.771283401883407,8.164965809277259,-2.300600867329175>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5 }
    cylinder {  m*<-3.771283401883407,-8.164965809277259,-2.3006008673291785>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5}

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
    sphere { m*<1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 1 }        
    sphere {  m*<1.3833405699944243,1.2743903680116328e-18,3.679662361473433>, 1 }
    sphere {  m*<4.5730155958237475,7.144676479647779e-18,-0.7980479722731812>, 1 }
    sphere {  m*<-3.771283401883407,8.164965809277259,-2.300600867329175>, 1}
    sphere { m*<-3.771283401883407,-8.164965809277259,-2.3006008673291785>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3833405699944243,1.2743903680116328e-18,3.679662361473433>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5 }
    cylinder { m*<4.5730155958237475,7.144676479647779e-18,-0.7980479722731812>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5}
    cylinder { m*<-3.771283401883407,8.164965809277259,-2.300600867329175>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5 }
    cylinder {  m*<-3.771283401883407,-8.164965809277259,-2.3006008673291785>, <1.168942252053123,-1.8547865507954763e-20,0.6873245115234791>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    