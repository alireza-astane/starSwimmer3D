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
    sphere { m*<-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 1 }        
    sphere {  m*<0.44370912562291587,0.24201322836874395,7.1507885689989426>, 1 }
    sphere {  m*<2.4956125175468222,-0.021017481323677326,-2.552122138940488>, 1 }
    sphere {  m*<-1.8607112363523246,2.2054224877085478,-2.2968583789052746>, 1}
    sphere { m*<-1.5929240153144928,-2.6822694546953496,-2.107312093742702>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44370912562291587,0.24201322836874395,7.1507885689989426>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5 }
    cylinder { m*<2.4956125175468222,-0.021017481323677326,-2.552122138940488>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5}
    cylinder { m*<-1.8607112363523246,2.2054224877085478,-2.2968583789052746>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5 }
    cylinder {  m*<-1.5929240153144928,-2.6822694546953496,-2.107312093742702>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5}

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
    sphere { m*<-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 1 }        
    sphere {  m*<0.44370912562291587,0.24201322836874395,7.1507885689989426>, 1 }
    sphere {  m*<2.4956125175468222,-0.021017481323677326,-2.552122138940488>, 1 }
    sphere {  m*<-1.8607112363523246,2.2054224877085478,-2.2968583789052746>, 1}
    sphere { m*<-1.5929240153144928,-2.6822694546953496,-2.107312093742702>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44370912562291587,0.24201322836874395,7.1507885689989426>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5 }
    cylinder { m*<2.4956125175468222,-0.021017481323677326,-2.552122138940488>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5}
    cylinder { m*<-1.8607112363523246,2.2054224877085478,-2.2968583789052746>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5 }
    cylinder {  m*<-1.5929240153144928,-2.6822694546953496,-2.107312093742702>, <-0.23909587645943473,-0.12305145671005155,-1.3229126134893072>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    