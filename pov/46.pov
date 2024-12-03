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
    sphere { m*<-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 1 }        
    sphere {  m*<-3.374497734000894e-18,-6.356032409327092e-19,9.731777195246273>, 1 }
    sphere {  m*<9.428090415820634,-3.5360573940049517e-19,-3.2725561380870567>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2725561380870567>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2725561380870567>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.374497734000894e-18,-6.356032409327092e-19,9.731777195246273>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5 }
    cylinder { m*<9.428090415820634,-3.5360573940049517e-19,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5}

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
    sphere { m*<-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 1 }        
    sphere {  m*<-3.374497734000894e-18,-6.356032409327092e-19,9.731777195246273>, 1 }
    sphere {  m*<9.428090415820634,-3.5360573940049517e-19,-3.2725561380870567>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2725561380870567>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2725561380870567>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.374497734000894e-18,-6.356032409327092e-19,9.731777195246273>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5 }
    cylinder { m*<9.428090415820634,-3.5360573940049517e-19,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2725561380870567>, <-1.1306600177177073e-18,1.1866130249169037e-18,0.0607771952462774>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    