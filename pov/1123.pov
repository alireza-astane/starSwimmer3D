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
    sphere { m*<0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 1 }        
    sphere {  m*<0.19636874281936725,-3.127999442414698e-18,4.134000171566697>, 1 }
    sphere {  m*<8.759652199943547,3.3472251520438348e-18,-1.9756598916016912>, 1 }
    sphere {  m*<-4.565932927088549,8.164965809277259,-2.163129770392908>, 1}
    sphere { m*<-4.565932927088549,-8.164965809277259,-2.163129770392911>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19636874281936725,-3.127999442414698e-18,4.134000171566697>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5 }
    cylinder { m*<8.759652199943547,3.3472251520438348e-18,-1.9756598916016912>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5}
    cylinder { m*<-4.565932927088549,8.164965809277259,-2.163129770392908>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5 }
    cylinder {  m*<-4.565932927088549,-8.164965809277259,-2.163129770392911>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5}

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
    sphere { m*<0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 1 }        
    sphere {  m*<0.19636874281936725,-3.127999442414698e-18,4.134000171566697>, 1 }
    sphere {  m*<8.759652199943547,3.3472251520438348e-18,-1.9756598916016912>, 1 }
    sphere {  m*<-4.565932927088549,8.164965809277259,-2.163129770392908>, 1}
    sphere { m*<-4.565932927088549,-8.164965809277259,-2.163129770392911>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19636874281936725,-3.127999442414698e-18,4.134000171566697>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5 }
    cylinder { m*<8.759652199943547,3.3472251520438348e-18,-1.9756598916016912>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5}
    cylinder { m*<-4.565932927088549,8.164965809277259,-2.163129770392908>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5 }
    cylinder {  m*<-4.565932927088549,-8.164965809277259,-2.163129770392911>, <0.1734457580792512,-4.337110261241907e-18,1.1340870433176518>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    