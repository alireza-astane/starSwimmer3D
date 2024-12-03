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
    sphere { m*<-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 1 }        
    sphere {  m*<0.5227480637866677,-0.29789445060477016,9.187050360029286>, 1 }
    sphere {  m*<7.87809950178664,-0.38681472659912636,-5.392442930016047>, 1 }
    sphere {  m*<-6.551549121192429,5.55776281313255,-3.5633554297736247>, 1}
    sphere { m*<-2.1333639282038526,-3.8257483211937724,-1.2769428890614558>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227480637866677,-0.29789445060477016,9.187050360029286>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5 }
    cylinder { m*<7.87809950178664,-0.38681472659912636,-5.392442930016047>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5}
    cylinder { m*<-6.551549121192429,5.55776281313255,-3.5633554297736247>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5 }
    cylinder {  m*<-2.1333639282038526,-3.8257483211937724,-1.2769428890614558>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5}

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
    sphere { m*<-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 1 }        
    sphere {  m*<0.5227480637866677,-0.29789445060477016,9.187050360029286>, 1 }
    sphere {  m*<7.87809950178664,-0.38681472659912636,-5.392442930016047>, 1 }
    sphere {  m*<-6.551549121192429,5.55776281313255,-3.5633554297736247>, 1}
    sphere { m*<-2.1333639282038526,-3.8257483211937724,-1.2769428890614558>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227480637866677,-0.29789445060477016,9.187050360029286>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5 }
    cylinder { m*<7.87809950178664,-0.38681472659912636,-5.392442930016047>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5}
    cylinder { m*<-6.551549121192429,5.55776281313255,-3.5633554297736247>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5 }
    cylinder {  m*<-2.1333639282038526,-3.8257483211937724,-1.2769428890614558>, <-0.904581961974426,-1.156653659158976,-0.6733064052554365>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    