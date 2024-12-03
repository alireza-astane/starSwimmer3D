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
    sphere { m*<0.11503939670123176,0.425343888762997,-0.06139735666468796>, 1 }        
    sphere {  m*<0.35577450144292333,0.5540539669433224,2.9261574144558615>, 1 }
    sphere {  m*<2.849747790707489,0.5273778641493714,-1.2906068821158727>, 1 }
    sphere {  m*<-1.506575963191659,2.7538178331815977,-1.0353431220806586>, 1}
    sphere { m*<-2.6603082879516338,-4.821053007106852,-1.6694162911841417>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.35577450144292333,0.5540539669433224,2.9261574144558615>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5 }
    cylinder { m*<2.849747790707489,0.5273778641493714,-1.2906068821158727>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5}
    cylinder { m*<-1.506575963191659,2.7538178331815977,-1.0353431220806586>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5 }
    cylinder {  m*<-2.6603082879516338,-4.821053007106852,-1.6694162911841417>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5}

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
    sphere { m*<0.11503939670123176,0.425343888762997,-0.06139735666468796>, 1 }        
    sphere {  m*<0.35577450144292333,0.5540539669433224,2.9261574144558615>, 1 }
    sphere {  m*<2.849747790707489,0.5273778641493714,-1.2906068821158727>, 1 }
    sphere {  m*<-1.506575963191659,2.7538178331815977,-1.0353431220806586>, 1}
    sphere { m*<-2.6603082879516338,-4.821053007106852,-1.6694162911841417>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.35577450144292333,0.5540539669433224,2.9261574144558615>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5 }
    cylinder { m*<2.849747790707489,0.5273778641493714,-1.2906068821158727>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5}
    cylinder { m*<-1.506575963191659,2.7538178331815977,-1.0353431220806586>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5 }
    cylinder {  m*<-2.6603082879516338,-4.821053007106852,-1.6694162911841417>, <0.11503939670123176,0.425343888762997,-0.06139735666468796>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    