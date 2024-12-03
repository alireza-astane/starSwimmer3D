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
    sphere { m*<-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 1 }        
    sphere {  m*<0.4264416918966638,0.2327811192948961,6.936497412463108>, 1 }
    sphere {  m*<2.5008136935964616,-0.01823664976736148,-2.487574850171736>, 1 }
    sphere {  m*<-1.8555100603026853,2.2082033192648636,-2.232311090136523>, 1}
    sphere { m*<-1.5877228392648535,-2.679488623139034,-2.0427648049739497>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4264416918966638,0.2327811192948961,6.936497412463108>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5 }
    cylinder { m*<2.5008136935964616,-0.01823664976736148,-2.487574850171736>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5}
    cylinder { m*<-1.8555100603026853,2.2082033192648636,-2.232311090136523>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5 }
    cylinder {  m*<-1.5877228392648535,-2.679488623139034,-2.0427648049739497>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5}

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
    sphere { m*<-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 1 }        
    sphere {  m*<0.4264416918966638,0.2327811192948961,6.936497412463108>, 1 }
    sphere {  m*<2.5008136935964616,-0.01823664976736148,-2.487574850171736>, 1 }
    sphere {  m*<-1.8555100603026853,2.2082033192648636,-2.232311090136523>, 1}
    sphere { m*<-1.5877228392648535,-2.679488623139034,-2.0427648049739497>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4264416918966638,0.2327811192948961,6.936497412463108>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5 }
    cylinder { m*<2.5008136935964616,-0.01823664976736148,-2.487574850171736>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5}
    cylinder { m*<-1.8555100603026853,2.2082033192648636,-2.232311090136523>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5 }
    cylinder {  m*<-1.5877228392648535,-2.679488623139034,-2.0427648049739497>, <-0.2338947004097955,-0.12027062515373568,-1.2583653247205557>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    