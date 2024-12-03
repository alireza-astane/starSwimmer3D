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
    sphere { m*<2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 1 }        
    sphere {  m*<3.804248483697605e-19,-4.4251714982368244e-18,5.778827679498097>, 1 }
    sphere {  m*<9.428090415820634,-8.281981439362519e-20,-2.430505653835259>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.430505653835259>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.430505653835259>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<3.804248483697605e-19,-4.4251714982368244e-18,5.778827679498097>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5 }
    cylinder { m*<9.428090415820634,-8.281981439362519e-20,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5}

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
    sphere { m*<2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 1 }        
    sphere {  m*<3.804248483697605e-19,-4.4251714982368244e-18,5.778827679498097>, 1 }
    sphere {  m*<9.428090415820634,-8.281981439362519e-20,-2.430505653835259>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.430505653835259>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.430505653835259>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<3.804248483697605e-19,-4.4251714982368244e-18,5.778827679498097>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5 }
    cylinder { m*<9.428090415820634,-8.281981439362519e-20,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.430505653835259>, <2.9883640743094867e-18,-5.1939655032771356e-18,0.9028276794980727>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    