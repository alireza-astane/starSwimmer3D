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
    sphere { m*<-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 1 }        
    sphere {  m*<0.3685968537669584,-0.17008023334800038,9.108496660718277>, 1 }
    sphere {  m*<7.72394829176693,-0.2590005093423569,-5.470996629327063>, 1 }
    sphere {  m*<-5.861885815238653,4.888415651854087,-3.21131496811799>, 1}
    sphere { m*<-2.3256499709328993,-3.6091893471116383,-1.3752787543631582>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3685968537669584,-0.17008023334800038,9.108496660718277>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5 }
    cylinder { m*<7.72394829176693,-0.2590005093423569,-5.470996629327063>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5}
    cylinder { m*<-5.861885815238653,4.888415651854087,-3.21131496811799>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5 }
    cylinder {  m*<-2.3256499709328993,-3.6091893471116383,-1.3752787543631582>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5}

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
    sphere { m*<-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 1 }        
    sphere {  m*<0.3685968537669584,-0.17008023334800038,9.108496660718277>, 1 }
    sphere {  m*<7.72394829176693,-0.2590005093423569,-5.470996629327063>, 1 }
    sphere {  m*<-5.861885815238653,4.888415651854087,-3.21131496811799>, 1}
    sphere { m*<-2.3256499709328993,-3.6091893471116383,-1.3752787543631582>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3685968537669584,-0.17008023334800038,9.108496660718277>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5 }
    cylinder { m*<7.72394829176693,-0.2590005093423569,-5.470996629327063>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5}
    cylinder { m*<-5.861885815238653,4.888415651854087,-3.21131496811799>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5 }
    cylinder {  m*<-2.3256499709328993,-3.6091893471116383,-1.3752787543631582>, <-1.0672434389334091,-0.9574017298534867,-0.7565902713074906>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    