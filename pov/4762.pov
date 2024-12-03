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
    sphere { m*<-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 1 }        
    sphere {  m*<0.4333537837977507,0.23647669950507388,7.022277395235764>, 1 }
    sphere {  m*<2.4987383415871918,-0.01934624580924916,-2.513330245520915>, 1 }
    sphere {  m*<-1.857585412311955,2.207093723222976,-2.2580664854857018>, 1}
    sphere { m*<-1.5897981912741233,-2.6805982191809212,-2.068520200323129>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4333537837977507,0.23647669950507388,7.022277395235764>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5 }
    cylinder { m*<2.4987383415871918,-0.01934624580924916,-2.513330245520915>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5}
    cylinder { m*<-1.857585412311955,2.207093723222976,-2.2580664854857018>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5 }
    cylinder {  m*<-1.5897981912741233,-2.6805982191809212,-2.068520200323129>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5}

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
    sphere { m*<-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 1 }        
    sphere {  m*<0.4333537837977507,0.23647669950507388,7.022277395235764>, 1 }
    sphere {  m*<2.4987383415871918,-0.01934624580924916,-2.513330245520915>, 1 }
    sphere {  m*<-1.857585412311955,2.207093723222976,-2.2580664854857018>, 1}
    sphere { m*<-1.5897981912741233,-2.6805982191809212,-2.068520200323129>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4333537837977507,0.23647669950507388,7.022277395235764>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5 }
    cylinder { m*<2.4987383415871918,-0.01934624580924916,-2.513330245520915>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5}
    cylinder { m*<-1.857585412311955,2.207093723222976,-2.2580664854857018>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5 }
    cylinder {  m*<-1.5897981912741233,-2.6805982191809212,-2.068520200323129>, <-0.2359700524190652,-0.12138022119562339,-1.284120720069735>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    