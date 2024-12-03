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
    sphere { m*<-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 1 }        
    sphere {  m*<0.3647671271101959,0.19980654309454743,6.171107834228133>, 1 }
    sphere {  m*<2.518902693514407,-0.008565286730629229,-2.2630879475754404>, 1 }
    sphere {  m*<-1.83742106038474,2.217874682301596,-2.007824187540227>, 1}
    sphere { m*<-1.5696338393469083,-2.6698172601023016,-1.8182779023776545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3647671271101959,0.19980654309454743,6.171107834228133>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5 }
    cylinder { m*<2.518902693514407,-0.008565286730629229,-2.2630879475754404>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5}
    cylinder { m*<-1.83742106038474,2.217874682301596,-2.007824187540227>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5 }
    cylinder {  m*<-1.5696338393469083,-2.6698172601023016,-1.8182779023776545>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5}

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
    sphere { m*<-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 1 }        
    sphere {  m*<0.3647671271101959,0.19980654309454743,6.171107834228133>, 1 }
    sphere {  m*<2.518902693514407,-0.008565286730629229,-2.2630879475754404>, 1 }
    sphere {  m*<-1.83742106038474,2.217874682301596,-2.007824187540227>, 1}
    sphere { m*<-1.5696338393469083,-2.6698172601023016,-1.8182779023776545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3647671271101959,0.19980654309454743,6.171107834228133>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5 }
    cylinder { m*<2.518902693514407,-0.008565286730629229,-2.2630879475754404>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5}
    cylinder { m*<-1.83742106038474,2.217874682301596,-2.007824187540227>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5 }
    cylinder {  m*<-1.5696338393469083,-2.6698172601023016,-1.8182779023776545>, <-0.21580570049185005,-0.11059926211700347,-1.0338784221242598>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    