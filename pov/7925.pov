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
    sphere { m*<-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 1 }        
    sphere {  m*<1.0648144254068397,0.6976868879388021,9.435629596917401>, 1 }
    sphere {  m*<8.43260162372963,0.4125946371465392,-5.135047832156521>, 1 }
    sphere {  m*<-6.463361569959359,6.9356760107671755,-3.6442409289749147>, 1}
    sphere { m*<-4.239236265438565,-8.752769074428594,-2.2127015392449207>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0648144254068397,0.6976868879388021,9.435629596917401>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5 }
    cylinder { m*<8.43260162372963,0.4125946371465392,-5.135047832156521>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5}
    cylinder { m*<-6.463361569959359,6.9356760107671755,-3.6442409289749147>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5 }
    cylinder {  m*<-4.239236265438565,-8.752769074428594,-2.2127015392449207>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5}

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
    sphere { m*<-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 1 }        
    sphere {  m*<1.0648144254068397,0.6976868879388021,9.435629596917401>, 1 }
    sphere {  m*<8.43260162372963,0.4125946371465392,-5.135047832156521>, 1 }
    sphere {  m*<-6.463361569959359,6.9356760107671755,-3.6442409289749147>, 1}
    sphere { m*<-4.239236265438565,-8.752769074428594,-2.2127015392449207>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0648144254068397,0.6976868879388021,9.435629596917401>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5 }
    cylinder { m*<8.43260162372963,0.4125946371465392,-5.135047832156521>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5}
    cylinder { m*<-6.463361569959359,6.9356760107671755,-3.6442409289749147>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5 }
    cylinder {  m*<-4.239236265438565,-8.752769074428594,-2.2127015392449207>, <-0.35435306879332096,-0.292252025941115,-0.413660500117739>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    