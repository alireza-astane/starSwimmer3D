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
    sphere { m*<-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 1 }        
    sphere {  m*<0.17737232809693132,0.09961509300477381,3.8455133743923895>, 1 }
    sphere {  m*<2.567428297148,0.017379139796315737,-1.6608787317130467>, 1 }
    sphere {  m*<-1.788895456751147,2.24381910882854,-1.4056149716778334>, 1}
    sphere { m*<-1.5211082357133152,-2.643872833575357,-1.2160686865152608>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17737232809693132,0.09961509300477381,3.8455133743923895>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5 }
    cylinder { m*<2.567428297148,0.017379139796315737,-1.6608787317130467>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5}
    cylinder { m*<-1.788895456751147,2.24381910882854,-1.4056149716778334>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5 }
    cylinder {  m*<-1.5211082357133152,-2.643872833575357,-1.2160686865152608>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5}

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
    sphere { m*<-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 1 }        
    sphere {  m*<0.17737232809693132,0.09961509300477381,3.8455133743923895>, 1 }
    sphere {  m*<2.567428297148,0.017379139796315737,-1.6608787317130467>, 1 }
    sphere {  m*<-1.788895456751147,2.24381910882854,-1.4056149716778334>, 1}
    sphere { m*<-1.5211082357133152,-2.643872833575357,-1.2160686865152608>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17737232809693132,0.09961509300477381,3.8455133743923895>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5 }
    cylinder { m*<2.567428297148,0.017379139796315737,-1.6608787317130467>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5}
    cylinder { m*<-1.788895456751147,2.24381910882854,-1.4056149716778334>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5 }
    cylinder {  m*<-1.5211082357133152,-2.643872833575357,-1.2160686865152608>, <-0.1672800968582569,-0.08465483559005839,-0.43166920626186367>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    