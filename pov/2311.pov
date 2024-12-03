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
    sphere { m*<1.0375388386113638,0.41153686152882374,0.4793292726738303>, 1 }        
    sphere {  m*<1.2815187405475739,0.44436109402031193,3.469210220075575>, 1 }
    sphere {  m*<3.7747659296101084,0.4443610940203118,-0.7480719884150435>, 1 }
    sphere {  m*<-2.950336394853425,6.706537905454446,-1.8785597959999032>, 1}
    sphere { m*<-3.7977282648528186,-7.86137217643224,-2.3789185674371787>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2815187405475739,0.44436109402031193,3.469210220075575>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5 }
    cylinder { m*<3.7747659296101084,0.4443610940203118,-0.7480719884150435>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5}
    cylinder { m*<-2.950336394853425,6.706537905454446,-1.8785597959999032>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5 }
    cylinder {  m*<-3.7977282648528186,-7.86137217643224,-2.3789185674371787>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5}

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
    sphere { m*<1.0375388386113638,0.41153686152882374,0.4793292726738303>, 1 }        
    sphere {  m*<1.2815187405475739,0.44436109402031193,3.469210220075575>, 1 }
    sphere {  m*<3.7747659296101084,0.4443610940203118,-0.7480719884150435>, 1 }
    sphere {  m*<-2.950336394853425,6.706537905454446,-1.8785597959999032>, 1}
    sphere { m*<-3.7977282648528186,-7.86137217643224,-2.3789185674371787>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2815187405475739,0.44436109402031193,3.469210220075575>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5 }
    cylinder { m*<3.7747659296101084,0.4443610940203118,-0.7480719884150435>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5}
    cylinder { m*<-2.950336394853425,6.706537905454446,-1.8785597959999032>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5 }
    cylinder {  m*<-3.7977282648528186,-7.86137217643224,-2.3789185674371787>, <1.0375388386113638,0.41153686152882374,0.4793292726738303>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    