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
    sphere { m*<-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 1 }        
    sphere {  m*<0.1612762655520572,0.0016075503313506267,9.002849746304747>, 1 }
    sphere {  m*<7.5166277035520315,-0.0873127256630061,-5.576643543740602>, 1 }
    sphere {  m*<-4.87387326153616,3.8963123664663164,-2.7067784854811263>, 1}
    sphere { m*<-2.5947481039026083,-3.287345443939403,-1.5130113781718242>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1612762655520572,0.0016075503313506267,9.002849746304747>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5 }
    cylinder { m*<7.5166277035520315,-0.0873127256630061,-5.576643543740602>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5}
    cylinder { m*<-4.87387326153616,3.8963123664663164,-2.7067784854811263>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5 }
    cylinder {  m*<-2.5947481039026083,-3.287345443939403,-1.5130113781718242>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5}

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
    sphere { m*<-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 1 }        
    sphere {  m*<0.1612762655520572,0.0016075503313506267,9.002849746304747>, 1 }
    sphere {  m*<7.5166277035520315,-0.0873127256630061,-5.576643543740602>, 1 }
    sphere {  m*<-4.87387326153616,3.8963123664663164,-2.7067784854811263>, 1}
    sphere { m*<-2.5947481039026083,-3.287345443939403,-1.5130113781718242>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1612762655520572,0.0016075503313506267,9.002849746304747>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5 }
    cylinder { m*<7.5166277035520315,-0.0873127256630061,-5.576643543740602>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5}
    cylinder { m*<-4.87387326153616,3.8963123664663164,-2.7067784854811263>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5 }
    cylinder {  m*<-2.5947481039026083,-3.287345443939403,-1.5130113781718242>, <-1.2870556597629454,-0.665539250488483,-0.8692738671567521>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    