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
    sphere { m*<-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 1 }        
    sphere {  m*<0.2074470367798804,0.27366216072616145,2.840217417457876>, 1 }
    sphere {  m*<2.7014203260444516,0.24698605793221062,-1.376546879113862>, 1 }
    sphere {  m*<-1.6549034278547032,2.4734260269644386,-1.1212831190786474>, 1}
    sphere { m*<-2.0694455887998093,-3.70411178642404,-1.3270741719637762>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2074470367798804,0.27366216072616145,2.840217417457876>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5 }
    cylinder { m*<2.7014203260444516,0.24698605793221062,-1.376546879113862>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5}
    cylinder { m*<-1.6549034278547032,2.4734260269644386,-1.1212831190786474>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5 }
    cylinder {  m*<-2.0694455887998093,-3.70411178642404,-1.3270741719637762>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5}

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
    sphere { m*<-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 1 }        
    sphere {  m*<0.2074470367798804,0.27366216072616145,2.840217417457876>, 1 }
    sphere {  m*<2.7014203260444516,0.24698605793221062,-1.376546879113862>, 1 }
    sphere {  m*<-1.6549034278547032,2.4734260269644386,-1.1212831190786474>, 1}
    sphere { m*<-2.0694455887998093,-3.70411178642404,-1.3270741719637762>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2074470367798804,0.27366216072616145,2.840217417457876>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5 }
    cylinder { m*<2.7014203260444516,0.24698605793221062,-1.376546879113862>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5}
    cylinder { m*<-1.6549034278547032,2.4734260269644386,-1.1212831190786474>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5 }
    cylinder {  m*<-2.0694455887998093,-3.70411178642404,-1.3270741719637762>, <-0.033288067961811274,0.14495208254583591,-0.1473373536626752>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    