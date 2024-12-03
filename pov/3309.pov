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
    sphere { m*<0.2820592212686768,0.7410709214533064,0.035372875702851025>, 1 }        
    sphere {  m*<0.5227943260103683,0.869780999633632,3.0229276468234008>, 1 }
    sphere {  m*<3.0167676152749325,0.8431048968396809,-1.1938366497483317>, 1 }
    sphere {  m*<-1.3395561386242139,3.069544865871909,-0.9385728897131179>, 1}
    sphere { m*<-3.251432221143025,-5.938488053288314,-2.0119097677585067>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227943260103683,0.869780999633632,3.0229276468234008>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5 }
    cylinder { m*<3.0167676152749325,0.8431048968396809,-1.1938366497483317>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5}
    cylinder { m*<-1.3395561386242139,3.069544865871909,-0.9385728897131179>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5 }
    cylinder {  m*<-3.251432221143025,-5.938488053288314,-2.0119097677585067>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5}

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
    sphere { m*<0.2820592212686768,0.7410709214533064,0.035372875702851025>, 1 }        
    sphere {  m*<0.5227943260103683,0.869780999633632,3.0229276468234008>, 1 }
    sphere {  m*<3.0167676152749325,0.8431048968396809,-1.1938366497483317>, 1 }
    sphere {  m*<-1.3395561386242139,3.069544865871909,-0.9385728897131179>, 1}
    sphere { m*<-3.251432221143025,-5.938488053288314,-2.0119097677585067>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227943260103683,0.869780999633632,3.0229276468234008>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5 }
    cylinder { m*<3.0167676152749325,0.8431048968396809,-1.1938366497483317>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5}
    cylinder { m*<-1.3395561386242139,3.069544865871909,-0.9385728897131179>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5 }
    cylinder {  m*<-3.251432221143025,-5.938488053288314,-2.0119097677585067>, <0.2820592212686768,0.7410709214533064,0.035372875702851025>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    