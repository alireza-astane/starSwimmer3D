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
    sphere { m*<-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 1 }        
    sphere {  m*<0.22243237337422733,0.12370664357722902,4.404714548643815>, 1 }
    sphere {  m*<2.5568785332299657,0.011738662309115938,-1.7918027095904663>, 1 }
    sphere {  m*<-1.7994452206691816,2.2381786313413405,-1.5365389495552528>, 1}
    sphere { m*<-1.5316579996313497,-2.649513311062557,-1.3469926643926804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22243237337422733,0.12370664357722902,4.404714548643815>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5 }
    cylinder { m*<2.5568785332299657,0.011738662309115938,-1.7918027095904663>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5}
    cylinder { m*<-1.7994452206691816,2.2381786313413405,-1.5365389495552528>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5 }
    cylinder {  m*<-1.5316579996313497,-2.649513311062557,-1.3469926643926804>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5}

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
    sphere { m*<-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 1 }        
    sphere {  m*<0.22243237337422733,0.12370664357722902,4.404714548643815>, 1 }
    sphere {  m*<2.5568785332299657,0.011738662309115938,-1.7918027095904663>, 1 }
    sphere {  m*<-1.7994452206691816,2.2381786313413405,-1.5365389495552528>, 1}
    sphere { m*<-1.5316579996313497,-2.649513311062557,-1.3469926643926804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22243237337422733,0.12370664357722902,4.404714548643815>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5 }
    cylinder { m*<2.5568785332299657,0.011738662309115938,-1.7918027095904663>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5}
    cylinder { m*<-1.7994452206691816,2.2381786313413405,-1.5365389495552528>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5 }
    cylinder {  m*<-1.5316579996313497,-2.649513311062557,-1.3469926643926804>, <-0.17782986077629154,-0.09029531307725824,-0.5625931841392842>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    