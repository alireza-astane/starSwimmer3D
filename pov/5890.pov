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
    sphere { m*<-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 1 }        
    sphere {  m*<-0.034614395075087595,0.2787676629351192,8.77758506723147>, 1 }
    sphere {  m*<6.738933126740521,0.10094079756096974,-5.399656678636062>, 1 }
    sphere {  m*<-3.1105278138717147,2.148747436584197,-1.9947752216353196>, 1}
    sphere { m*<-2.8427405928338834,-2.7389445058197004,-1.8052289364727492>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.034614395075087595,0.2787676629351192,8.77758506723147>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5 }
    cylinder { m*<6.738933126740521,0.10094079756096974,-5.399656678636062>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5}
    cylinder { m*<-3.1105278138717147,2.148747436584197,-1.9947752216353196>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5 }
    cylinder {  m*<-2.8427405928338834,-2.7389445058197004,-1.8052289364727492>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5}

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
    sphere { m*<-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 1 }        
    sphere {  m*<-0.034614395075087595,0.2787676629351192,8.77758506723147>, 1 }
    sphere {  m*<6.738933126740521,0.10094079756096974,-5.399656678636062>, 1 }
    sphere {  m*<-3.1105278138717147,2.148747436584197,-1.9947752216353196>, 1}
    sphere { m*<-2.8427405928338834,-2.7389445058197004,-1.8052289364727492>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.034614395075087595,0.2787676629351192,8.77758506723147>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5 }
    cylinder { m*<6.738933126740521,0.10094079756096974,-5.399656678636062>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5}
    cylinder { m*<-3.1105278138717147,2.148747436584197,-1.9947752216353196>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5 }
    cylinder {  m*<-2.8427405928338834,-2.7389445058197004,-1.8052289364727492>, <-1.438246060034454,-0.18050909088861664,-1.1127174984230872>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    