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
    sphere { m*<-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 1 }        
    sphere {  m*<-7.405935577760787e-18,-2.2615427023044614e-18,8.70161388106062>, 1 }
    sphere {  m*<9.428090415820634,-1.9496734889934786e-18,-3.042719452272725>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.042719452272725>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.042719452272725>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.405935577760787e-18,-2.2615427023044614e-18,8.70161388106062>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5 }
    cylinder { m*<9.428090415820634,-1.9496734889934786e-18,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5}

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
    sphere { m*<-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 1 }        
    sphere {  m*<-7.405935577760787e-18,-2.2615427023044614e-18,8.70161388106062>, 1 }
    sphere {  m*<9.428090415820634,-1.9496734889934786e-18,-3.042719452272725>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.042719452272725>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.042719452272725>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.405935577760787e-18,-2.2615427023044614e-18,8.70161388106062>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5 }
    cylinder { m*<9.428090415820634,-1.9496734889934786e-18,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.042719452272725>, <-4.106915192861232e-18,1.120719504367041e-18,0.29061388106060765>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    