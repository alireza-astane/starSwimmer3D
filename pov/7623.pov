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
    sphere { m*<-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 1 }        
    sphere {  m*<0.9085926265784491,0.35746631361771253,9.363285226939633>, 1 }
    sphere {  m*<8.276379824901252,0.07237406282545122,-5.207392202134299>, 1 }
    sphere {  m*<-6.619583368787745,6.59545543644609,-3.7165852989526913>, 1}
    sphere { m*<-3.529328231140189,-7.206727990818603,-1.883951967362875>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9085926265784491,0.35746631361771253,9.363285226939633>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5 }
    cylinder { m*<8.276379824901252,0.07237406282545122,-5.207392202134299>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5}
    cylinder { m*<-6.619583368787745,6.59545543644609,-3.7165852989526913>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5 }
    cylinder {  m*<-3.529328231140189,-7.206727990818603,-1.883951967362875>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5}

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
    sphere { m*<-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 1 }        
    sphere {  m*<0.9085926265784491,0.35746631361771253,9.363285226939633>, 1 }
    sphere {  m*<8.276379824901252,0.07237406282545122,-5.207392202134299>, 1 }
    sphere {  m*<-6.619583368787745,6.59545543644609,-3.7165852989526913>, 1}
    sphere { m*<-3.529328231140189,-7.206727990818603,-1.883951967362875>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9085926265784491,0.35746631361771253,9.363285226939633>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5 }
    cylinder { m*<8.276379824901252,0.07237406282545122,-5.207392202134299>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5}
    cylinder { m*<-6.619583368787745,6.59545543644609,-3.7165852989526913>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5 }
    cylinder {  m*<-3.529328231140189,-7.206727990818603,-1.883951967362875>, <-0.5105748676217129,-0.6324726002622046,-0.4860048700955156>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    