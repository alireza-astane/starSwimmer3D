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
    sphere { m*<-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 1 }        
    sphere {  m*<0.8818539282210539,0.29923464953513346,9.350902868643933>, 1 }
    sphere {  m*<8.249641126543859,0.014142398742872375,-5.219774560429995>, 1 }
    sphere {  m*<-6.646322067145139,6.537223772363514,-3.728967657248388>, 1}
    sphere { m*<-3.404064011763416,-6.933926979833806,-1.825943666987806>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8818539282210539,0.29923464953513346,9.350902868643933>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5 }
    cylinder { m*<8.249641126543859,0.014142398742872375,-5.219774560429995>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5}
    cylinder { m*<-6.646322067145139,6.537223772363514,-3.728967657248388>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5 }
    cylinder {  m*<-3.404064011763416,-6.933926979833806,-1.825943666987806>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5}

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
    sphere { m*<-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 1 }        
    sphere {  m*<0.8818539282210539,0.29923464953513346,9.350902868643933>, 1 }
    sphere {  m*<8.249641126543859,0.014142398742872375,-5.219774560429995>, 1 }
    sphere {  m*<-6.646322067145139,6.537223772363514,-3.728967657248388>, 1}
    sphere { m*<-3.404064011763416,-6.933926979833806,-1.825943666987806>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8818539282210539,0.29923464953513346,9.350902868643933>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5 }
    cylinder { m*<8.249641126543859,0.014142398742872375,-5.219774560429995>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5}
    cylinder { m*<-6.646322067145139,6.537223772363514,-3.728967657248388>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5 }
    cylinder {  m*<-3.404064011763416,-6.933926979833806,-1.825943666987806>, <-0.5373135659791075,-0.6907042643447835,-0.49838722839121297>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    