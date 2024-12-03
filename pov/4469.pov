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
    sphere { m*<-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 1 }        
    sphere {  m*<0.305484074447356,0.16811059932602312,5.435397240546757>, 1 }
    sphere {  m*<2.5354511988452,0.0002824441276488704,-2.0577187883538444>, 1 }
    sphere {  m*<-1.820872555053947,2.226722413159874,-1.8024550283186311>, 1}
    sphere { m*<-1.5530853340161153,-2.6609695292440234,-1.6129087431560583>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.305484074447356,0.16811059932602312,5.435397240546757>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5 }
    cylinder { m*<2.5354511988452,0.0002824441276488704,-2.0577187883538444>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5}
    cylinder { m*<-1.820872555053947,2.226722413159874,-1.8024550283186311>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5 }
    cylinder {  m*<-1.5530853340161153,-2.6609695292440234,-1.6129087431560583>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5}

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
    sphere { m*<-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 1 }        
    sphere {  m*<0.305484074447356,0.16811059932602312,5.435397240546757>, 1 }
    sphere {  m*<2.5354511988452,0.0002824441276488704,-2.0577187883538444>, 1 }
    sphere {  m*<-1.820872555053947,2.226722413159874,-1.8024550283186311>, 1}
    sphere { m*<-1.5530853340161153,-2.6609695292440234,-1.6129087431560583>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.305484074447356,0.16811059932602312,5.435397240546757>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5 }
    cylinder { m*<2.5354511988452,0.0002824441276488704,-2.0577187883538444>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5}
    cylinder { m*<-1.820872555053947,2.226722413159874,-1.8024550283186311>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5 }
    cylinder {  m*<-1.5530853340161153,-2.6609695292440234,-1.6129087431560583>, <-0.19925719516105717,-0.10175153125872534,-0.8285092629026634>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    