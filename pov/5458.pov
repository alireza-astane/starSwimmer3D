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
    sphere { m*<-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 1 }        
    sphere {  m*<0.28192985770449647,0.2854003000131073,8.501421367355402>, 1 }
    sphere {  m*<4.73678198069564,0.03966874724931685,-4.152474562220806>, 1 }
    sphere {  m*<-2.4914514185650565,2.168851676208672,-2.350454077888932>, 1}
    sphere { m*<-2.223664197527225,-2.7188402661952256,-2.1609077927263614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28192985770449647,0.2854003000131073,8.501421367355402>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5 }
    cylinder { m*<4.73678198069564,0.03966874724931685,-4.152474562220806>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5}
    cylinder { m*<-2.4914514185650565,2.168851676208672,-2.350454077888932>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5 }
    cylinder {  m*<-2.223664197527225,-2.7188402661952256,-2.1609077927263614>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5}

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
    sphere { m*<-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 1 }        
    sphere {  m*<0.28192985770449647,0.2854003000131073,8.501421367355402>, 1 }
    sphere {  m*<4.73678198069564,0.03966874724931685,-4.152474562220806>, 1 }
    sphere {  m*<-2.4914514185650565,2.168851676208672,-2.350454077888932>, 1}
    sphere { m*<-2.223664197527225,-2.7188402661952256,-2.1609077927263614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28192985770449647,0.2854003000131073,8.501421367355402>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5 }
    cylinder { m*<4.73678198069564,0.03966874724931685,-4.152474562220806>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5}
    cylinder { m*<-2.4914514185650565,2.168851676208672,-2.350454077888932>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5 }
    cylinder {  m*<-2.223664197527225,-2.7188402661952256,-2.1609077927263614>, <-0.8420861632888383,-0.15998766283558247,-1.4252004552984725>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    