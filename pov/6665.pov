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
    sphere { m*<-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 1 }        
    sphere {  m*<0.33534792342793374,-0.14255566431642666,9.091553652059567>, 1 }
    sphere {  m*<7.6906993614279076,-0.2314759403107831,-5.487939637985775>, 1 }
    sphere {  m*<-5.708805538071006,4.737602301431892,-3.1331610311442413>, 1}
    sphere { m*<-2.3679157487572273,-3.5602576542939763,-1.3969017762465221>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33534792342793374,-0.14255566431642666,9.091553652059567>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5 }
    cylinder { m*<7.6906993614279076,-0.2314759403107831,-5.487939637985775>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5}
    cylinder { m*<-5.708805538071006,4.737602301431892,-3.1331610311442413>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5 }
    cylinder {  m*<-2.3679157487572273,-3.5602576542939763,-1.3969017762465221>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5}

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
    sphere { m*<-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 1 }        
    sphere {  m*<0.33534792342793374,-0.14255566431642666,9.091553652059567>, 1 }
    sphere {  m*<7.6906993614279076,-0.2314759403107831,-5.487939637985775>, 1 }
    sphere {  m*<-5.708805538071006,4.737602301431892,-3.1331610311442413>, 1}
    sphere { m*<-2.3679157487572273,-3.5602576542939763,-1.3969017762465221>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33534792342793374,-0.14255566431642666,9.091553652059567>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5 }
    cylinder { m*<7.6906993614279076,-0.2314759403107831,-5.487939637985775>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5}
    cylinder { m*<-5.708805538071006,4.737602301431892,-3.1331610311442413>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5 }
    cylinder {  m*<-2.3679157487572273,-3.5602576542939763,-1.3969017762465221>, <-1.1024187528457055,-0.9126911323338479,-0.7746102052669319>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    