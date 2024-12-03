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
    sphere { m*<-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 1 }        
    sphere {  m*<0.6337293041211963,-0.24113233348232188,9.235999444530929>, 1 }
    sphere {  m*<8.001516502444005,-0.5262245842745839,-5.334677984543012>, 1 }
    sphere {  m*<-6.8944466912449975,5.996856789346071,-3.8438710813614057>, 1}
    sphere { m*<-2.1464993564210375,-4.19519670982537,-1.2435811325696684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6337293041211963,-0.24113233348232188,9.235999444530929>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5 }
    cylinder { m*<8.001516502444005,-0.5262245842745839,-5.334677984543012>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5}
    cylinder { m*<-6.8944466912449975,5.996856789346071,-3.8438710813614057>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5 }
    cylinder {  m*<-2.1464993564210375,-4.19519670982537,-1.2435811325696684>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5}

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
    sphere { m*<-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 1 }        
    sphere {  m*<0.6337293041211963,-0.24113233348232188,9.235999444530929>, 1 }
    sphere {  m*<8.001516502444005,-0.5262245842745839,-5.334677984543012>, 1 }
    sphere {  m*<-6.8944466912449975,5.996856789346071,-3.8438710813614057>, 1}
    sphere { m*<-2.1464993564210375,-4.19519670982537,-1.2435811325696684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6337293041211963,-0.24113233348232188,9.235999444530929>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5 }
    cylinder { m*<8.001516502444005,-0.5262245842745839,-5.334677984543012>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5}
    cylinder { m*<-6.8944466912449975,5.996856789346071,-3.8438710813614057>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5 }
    cylinder {  m*<-2.1464993564210375,-4.19519670982537,-1.2435811325696684>, <-0.7854381900789668,-1.23107124736224,-0.6132906525042274>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    