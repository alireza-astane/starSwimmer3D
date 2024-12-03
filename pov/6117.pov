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
    sphere { m*<-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 1 }        
    sphere {  m*<-0.05718857824309831,0.1850152138020541,8.891508693342663>, 1 }
    sphere {  m*<7.2981628597568795,0.09609493780769696,-5.687984596702698>, 1 }
    sphere {  m*<-3.696404593726932,2.6311324256641755,-2.1049885212867725>, 1}
    sphere { m*<-2.896330276847893,-2.8875788739691433,-1.6676090870751894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05718857824309831,0.1850152138020541,8.891508693342663>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5 }
    cylinder { m*<7.2981628597568795,0.09609493780769696,-5.687984596702698>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5}
    cylinder { m*<-3.696404593726932,2.6311324256641755,-2.1049885212867725>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5 }
    cylinder {  m*<-2.896330276847893,-2.8875788739691433,-1.6676090870751894>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5}

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
    sphere { m*<-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 1 }        
    sphere {  m*<-0.05718857824309831,0.1850152138020541,8.891508693342663>, 1 }
    sphere {  m*<7.2981628597568795,0.09609493780769696,-5.687984596702698>, 1 }
    sphere {  m*<-3.696404593726932,2.6311324256641755,-2.1049885212867725>, 1}
    sphere { m*<-2.896330276847893,-2.8875788739691433,-1.6676090870751894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05718857824309831,0.1850152138020541,8.891508693342663>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5 }
    cylinder { m*<7.2981628597568795,0.09609493780769696,-5.687984596702698>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5}
    cylinder { m*<-3.696404593726932,2.6311324256641755,-2.1049885212867725>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5 }
    cylinder {  m*<-2.896330276847893,-2.8875788739691433,-1.6676090870751894>, <-1.5194232934964151,-0.31047125627398897,-0.9886781871903121>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    