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
    sphere { m*<-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 1 }        
    sphere {  m*<0.5167466181784179,-0.2929100165721438,9.183992040764702>, 1 }
    sphere {  m*<7.872098056178388,-0.3818302925665,-5.395501249280629>, 1 }
    sphere {  m*<-6.525243036708757,5.532499314117356,-3.549929054907801>, 1}
    sphere { m*<-2.1407470048715527,-3.817597696725485,-1.2807176196298082>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5167466181784179,-0.2929100165721438,9.183992040764702>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5 }
    cylinder { m*<7.872098056178388,-0.3818302925665,-5.395501249280629>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5}
    cylinder { m*<-6.525243036708757,5.532499314117356,-3.549929054907801>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5 }
    cylinder {  m*<-2.1407470048715527,-3.817597696725485,-1.2807176196298082>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5}

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
    sphere { m*<-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 1 }        
    sphere {  m*<0.5167466181784179,-0.2929100165721438,9.183992040764702>, 1 }
    sphere {  m*<7.872098056178388,-0.3818302925665,-5.395501249280629>, 1 }
    sphere {  m*<-6.525243036708757,5.532499314117356,-3.549929054907801>, 1}
    sphere { m*<-2.1407470048715527,-3.817597696725485,-1.2807176196298082>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5167466181784179,-0.2929100165721438,9.183992040764702>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5 }
    cylinder { m*<7.872098056178388,-0.3818302925665,-5.395501249280629>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5}
    cylinder { m*<-6.525243036708757,5.532499314117356,-3.549929054907801>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5 }
    cylinder {  m*<-2.1407470048715527,-3.817597696725485,-1.2807176196298082>, <-0.9109016715232877,-1.1491151513065545,-0.676540903405991>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    