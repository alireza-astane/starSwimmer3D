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
    sphere { m*<-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 1 }        
    sphere {  m*<-7.941291295763785e-18,-1.9039785661034675e-18,8.253622391812186>, 1 }
    sphere {  m*<9.428090415820634,-2.161639649858862e-18,-2.9447109415211585>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.9447109415211585>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.9447109415211585>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.941291295763785e-18,-1.9039785661034675e-18,8.253622391812186>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5 }
    cylinder { m*<9.428090415820634,-2.161639649858862e-18,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5}

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
    sphere { m*<-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 1 }        
    sphere {  m*<-7.941291295763785e-18,-1.9039785661034675e-18,8.253622391812186>, 1 }
    sphere {  m*<9.428090415820634,-2.161639649858862e-18,-2.9447109415211585>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.9447109415211585>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.9447109415211585>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.941291295763785e-18,-1.9039785661034675e-18,8.253622391812186>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5 }
    cylinder { m*<9.428090415820634,-2.161639649858862e-18,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.9447109415211585>, <-4.5093470673996276e-18,7.741517531604174e-19,0.3886223918121734>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    