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
    sphere { m*<-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 1 }        
    sphere {  m*<0.4350807294657042,0.237400018570542,7.043709021401363>, 1 }
    sphere {  m*<2.4982184262775564,-0.019624220806797314,-2.519782463885772>, 1 }
    sphere {  m*<-1.8581053276215906,2.206815748225427,-2.264518703850559>, 1}
    sphere { m*<-1.5903181065837588,-2.6808761941784702,-2.0749724186879863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4350807294657042,0.237400018570542,7.043709021401363>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5 }
    cylinder { m*<2.4982184262775564,-0.019624220806797314,-2.519782463885772>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5}
    cylinder { m*<-1.8581053276215906,2.206815748225427,-2.264518703850559>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5 }
    cylinder {  m*<-1.5903181065837588,-2.6808761941784702,-2.0749724186879863>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5}

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
    sphere { m*<-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 1 }        
    sphere {  m*<0.4350807294657042,0.237400018570542,7.043709021401363>, 1 }
    sphere {  m*<2.4982184262775564,-0.019624220806797314,-2.519782463885772>, 1 }
    sphere {  m*<-1.8581053276215906,2.206815748225427,-2.264518703850559>, 1}
    sphere { m*<-1.5903181065837588,-2.6808761941784702,-2.0749724186879863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4350807294657042,0.237400018570542,7.043709021401363>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5 }
    cylinder { m*<2.4982184262775564,-0.019624220806797314,-2.519782463885772>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5}
    cylinder { m*<-1.8581053276215906,2.206815748225427,-2.264518703850559>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5 }
    cylinder {  m*<-1.5903181065837588,-2.6808761941784702,-2.0749724186879863>, <-0.23648996772870073,-0.12165819619317157,-1.2905729384345923>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    